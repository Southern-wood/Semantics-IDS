import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.DCdetector import DCdetector
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
import warnings
warnings.filterwarnings('ignore')
import shutil
from sklearn.metrics import fbeta_score,precision_recall_fscore_support
import GPUtil
import random
from generate_testfiles import *
import pickle
import sys


def set_random_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def find_continuous_ones(arr):
    #find all 1 indexes
    ones_indices = np.where(arr == 1)[0]
    if len(ones_indices) == 0:
        return []
    
    #Initialize results
    result = []
    start = ones_indices[0]
    
    for i in range(1, len(ones_indices)):
        #If not continuous
        if ones_indices[i] != ones_indices[i - 1] + 1:
            end = ones_indices[i - 1]
            result.append((start, end))
            start = ones_indices[i]
    
    #add the last pair
    result.append((start, ones_indices[-1]))
    
    #Filter the single anomaly
    #final_result = []
    #for one_result in result:
    #    if one_result[1] - one_result[0] >2:
    #        final_result.append(one_result)

    return result

def compute_range_metrics(pred_array,gt_array,metric_name,display=False):
    
    #Filter single anomaly at first
    ones_indices = np.where(pred_array == 1)[0]
    new_pred = np.zeros_like(pred_array)
    if len(ones_indices) >= 2:
        for i in range(1, len(ones_indices)):
            if ones_indices[i] == ones_indices[i - 1] + 1:
                new_pred[ones_indices[i - 1]] = 1
                new_pred[ones_indices[i]] = 1
    #

    pred_ranges = find_continuous_ones(new_pred) #pred_array
    gt_ranges = find_continuous_ones(gt_array)

    detected_gts = []
    for gt_range in gt_ranges:
        for pred_range in pred_ranges:
            if gt_range[0] <= pred_range[1] and pred_range[0] <= gt_range[1]:
                detected_gts.append(gt_range)
                break
    
    detected_preds = []
    for pred_range in pred_ranges:
        for gt_range in gt_ranges:
             if pred_range[0] <= gt_range[1] and gt_range[0] <= pred_range[1]:
                detected_preds.append(pred_range)
                break
    
    latency_info = []
    for detected_gt in detected_gts:
        for i in range(detected_gt[0],pred_array.shape[0]):
            if pred_array[i] == 1:
                latency_info.append((detected_gt,i,i-detected_gt[0]))
                break
    
    avg_latency = None
    if len(latency_info) != 0:
        avg_latency = 0
        for one_latency in latency_info:
            avg_latency += one_latency[2]
        avg_latency = avg_latency/len(latency_info)

    TP = len(detected_gts)
    FP = len(pred_ranges) - len(detected_preds)
    range_recall = TP/len(gt_ranges)
    range_precision = 0
    if TP+FP != 0:
        range_precision = TP/(TP+FP)
    
    range_f1,range_f1_0_2,range_f1_0_5,range_f1_0_1 = 0,0,0,0

    if range_precision+range_recall != 0:
        range_f1 = 2*range_precision*range_recall/(range_precision+range_recall)
        range_f1_0_2 = (1+(0.2*0.2))*range_precision*range_recall/(((0.2*0.2)*range_precision)+range_recall)
        range_f1_0_5 = (1+(0.5*0.5))*range_precision*range_recall/(((0.5*0.5)*range_precision)+range_recall)
        range_f1_0_1 = (1+(0.1*0.1))*range_precision*range_recall/(((0.1*0.1)*range_precision)+range_recall)

    results = {"range_precision":range_precision,"range_recall":range_recall,"range_f1":range_f1,"range_f1_0_2":range_f1_0_2,"range_f1_0_5":range_f1_0_5,"range_f1_0_1":range_f1_0_1,"avg_latency":avg_latency}
    
    if display:
        print("predicted ranges")
        print(str(pred_ranges))
        print("gt ranges")
        print(str(gt_ranges))
        print("detected ranges")
        print(str(detected_preds))
        print("detected gt ranges")
        print(str(detected_gts))
        print(metric_name+": "+str(results[metric_name]))
        print("range_precision: "+str(range_precision))
        print("range_recall: "+str(range_recall))
        print("avg_latency: "+str(avg_latency))
        print("=====================================")
    else:
        return results
    
def compute_point_metrics(pred_array,gt_array,metric_name,display=False):
    #Prevent Original labels being adjusted
    pred = np.copy(pred_array)
    gt = np.copy(gt_array)

    #Adjust must be taken lastly
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    precision_result,recall_result,f1_score_result,support = precision_recall_fscore_support(gt,pred,average='binary')
    fbeta_score_0_5_result = fbeta_score(gt,pred,average='binary',beta=0.5)
    fbeta_score_0_2_result = fbeta_score(gt,pred,average='binary',beta=0.2)
    fbeta_score_0_1_result = fbeta_score(gt,pred,average='binary',beta=0.1)

    results = {"precision":precision_result,"recall":recall_result,"f1":f1_score_result,"fbeta_0_5":fbeta_score_0_5_result,"fbeta_0_2":fbeta_score_0_2_result,"fbeta_0_1":fbeta_score_0_1_result}

    if display:
        print(metric_name+": "+str(results[metric_name]))
        print("precision: "+str(precision_result))
        print("recall: "+str(recall_result))
        print("=====================================")
    else:
        return results

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0, quality_type='', level=''):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name
        self.quality_type = quality_type
        self.level = level

    def __call__(self, val_loss, val_loss2, model, path, epoch):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path, epoch)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path, epoch):
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_epoch' + str(epoch) +'_checkpoint.pth'))
        if self.quality_type == 'pure':
            model_path = os.path.join(path, str(self.dataset) + '_' + str(self.quality_type) + '_latest_checkpoint.pth')
        else:
            model_path = os.path.join(path, str(self.dataset) + '_' + str(self.quality_type) + '_' + str(self.level) + '_latest_checkpoint.pth')
        torch.save(model.state_dict(), model_path)
        model_path.replace('latest', 'epoch_' + str(epoch))
        torch.save(model.state_dict(), model_path)
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        attack_path = self.attack_path
        normal_path = self.normal_path
        labels_path = self.labels_path

        self.train_loader = get_loader_segment(normal_path, attack_path, labels_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(normal_path, attack_path, labels_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(normal_path, attack_path, labels_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(normal_path, attack_path, labels_path, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)

        # self.device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
        self.build_model()
        
        
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
        

    def build_model(self):
        self.model = DCdetector(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads, d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size, channel=self.input_c)
        
        if torch.cuda.is_available():
            self.model.cuda()
      
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            loss_1.append((prior_loss - series_loss).item())

        return np.average(loss_1), np.average(loss_2)


    def train(self):

        time_now = time.time()
        path = self.model_save_path
        
        # shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)


        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.dataset, quality_type=self.quality_type, level=self.level)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                series, prior = self.model(input)
                # list of numbers , and list len : batch_size * win_size * win_size
                
                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss = prior_loss - series_loss 
                # print(i)
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            early_stopping(vali_loss1, vali_loss2, self.model, path, epoch)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

            
    def test(self):
        if self.quality_type == 'pure':
            model_path = os.path.join(self.model_save_path, str(self.dataset) + '_' + str(self.quality_type) + '_latest_checkpoint.pth')
        else:
            model_path = os.path.join(self.model_save_path, str(self.dataset) + '_' + str(self.quality_type) + '_' + str(self.level) + '_latest_checkpoint.pth')
        self.model.load_state_dict( 
            torch.load(model_path))
        self.model.eval()
        temperature = 50

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((series_loss + prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        #change from thre_loader to test_loader
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((series_loss + prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        
        
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        TP = np.sum((pred == 1) & (gt == 1))
        FP = np.sum((pred == 1) & (gt == 0))
        print("True Positives (TP):", TP)
        print("False Positives (FP):", FP)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))

        return accuracy, precision, recall, f_score
