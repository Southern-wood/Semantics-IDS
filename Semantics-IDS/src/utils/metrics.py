import numpy as np

from .spot import SPOT, quikSPOT, dSPOT
from ..constants import lm, color
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """

    if len(predict) != len(actual):
        raise ValueError("predict and actual must have the same length")
    

    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    
    precision = 0
    if TP+FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = TP / (TP + FP + 0.00001)
    
    recall = 0
    if TP + FN != 0:
        recall = TP / (TP + FN)
    else:
        recall = TP / (TP + FN + 0.00001)

    f1 = 0
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 2 * precision * recall / (precision + recall + 0.00001)
    
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc

def after_adjust(predict,actual):
    anomaly_state = False
    for i in range(predict.shape[0]):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True

    return predict

def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t

def pot_eval(init_score, score, label, q=1e-5):
    # print(str(lm))
    min_lms = 1e-5
    lms = lm[0]
    while True and lms > min_lms:
        try:
            s = SPOT()  # SPOT object
            s.fit(init_score, score)  # data import
            s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step
            
        except Exception as e:
            # print(e)
            lms = lms * 0.999
            # print(lms)
        else: break
    if lms <= min_lms:
        return {}, np.array([0] * len(score))
    
    ret = s.run(dynamic=False)  # run
    # print(len(ret['alarms']))
    # print(len(ret['thresholds']))
    pot_th = np.mean(ret['thresholds']) * lm[1]
    score = np.asarray(score)
    predict = score > pot_th
    prediction_rate = np.sum(predict) / len(predict)
    # if prediction_rate > 0.5:
    #     raise ValueError("Prediction rate is too high")

    p_t = calc_point2point(predict, label)

    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    # DEBUG - np.save(f'{debug}.npy', np.array(pred))
    # DEBUG - print(np.argwhere(np.array(pred)))
    p_t_adjust = calc_point2point(pred, label)

    # print('POT result: ', p_t, pot_th, p_latency)
    return {
        'f1': p_t[0],
        'f1_adjusted': p_t_adjust[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
        'threshold': pot_th,
        # 'pot-latency': p_latency
    }, np.array(predict)

def quik_pot_eval(init_score, score, label, q=1e-5):
    # print(str(lm))
    lms = lm[0]
    min_lms = lm[1]
    while True and lms > min_lms:
        try:
            s = quikSPOT(q)  # SPOT object
            s.fit(init_score, score)  # data import
            s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step
        except Exception as e:
            # print(e)
            lms = lms * 0.999
            # print(lms)
        else: break
    if lms <= min_lms:
        return {}, np.array([0] * len(score))
    
    ret = s.run(dynamic=True)  # run
    # print(len(ret['alarms']))
    # print(len(ret['thresholds']))
    pot_th = np.mean(ret['thresholds']) * lm[1]
    score = np.asarray(score)
    predict = score > pot_th
    p_t = calc_point2point(predict, label)

    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    # DEBUG - np.save(f'{debug}.npy', np.array(pred))
    # DEBUG - print(np.argwhere(np.array(pred)))
    p_t_adjust = calc_point2point(pred, label)

    # print('POT result: ', p_t, pot_th, p_latency)
    return {
        'f1': p_t[0],
        'f1_adjusted': p_t_adjust[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
        'threshold': pot_th,
        # 'pot-latency': p_latency
    }, np.array(predict)

def eval_f1score(score, label):
    fi_score_list = []
    threholds = []
    for threhold in range(score.max()):
        predict = adjust_predicts(score, label, threhold)
        f1 = f1_score(label, predict)
        fi_score_list.append(f1)
        threholds.append(threhold)
    return fi_score_list, threholds

def eval_f1score_threshold(score, label, threshold, verbose=False):
    f1, precision, recall, accuracy, specificity, auc = 0, 0, 0, 0, 0, 0
    for i in range(threshold - 1, threshold + 2):
        prediction = adjust_predicts(score, label, i)
        tmp_f1 = f1_score(label, prediction)
        if tmp_f1 < f1: 
            f1 = tmp_f1
            precision = precision_score(label, prediction)
            recall = recall_score(label, prediction)
            accuracy = accuracy_score(label, prediction)
            tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()
            specificity = tn / (tn + fp)

    # Calculate ROC AUC if possible
    try:
      auc = roc_auc_score(label, score)
    except:
      auc = float('nan')  # In case of only one class being present

    if verbose:
        print(f"\n{color.BOLD}Evaluation Metrics with threshold {i}:{color.ENDC}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"F1 Score: {f1:.4f}")

    return f1