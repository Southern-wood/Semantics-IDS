import sys
import os

types = ['pure', 'noise', 'missing', 'duplicate', 'delay', 'mismatch']

# these functions are used to generate the path of the training and testing files

def generate_trainpath_and_label(prefix, dataset, type, level=None):
    trainpath = prefix + '/' + dataset + '/train/'
    labelpath = prefix + '/' + dataset + '/test/'
    
    if type == 'pure':
        trainpath = trainpath + dataset + '_pure.npy'
        labelpath = labelpath + dataset + '_pure_labels.npy'
    else:
        trainpath = trainpath + dataset + '_' + type + '_' + level + '.npy'
        labelpath = labelpath + dataset + '_' + type + '_' + level + '_labels.npy'

    return trainpath, labelpath

def generate_testfiles(prefix, dataset):
    testfiles = []
    testpath = prefix + '/' + dataset + '/test/'
    for type in types:
        if type == 'pure':
            testfiles.append(testpath + dataset + '_pure.npy')
        else:
            testfiles.append(testpath + dataset + '_' + type + '_low.npy')
            testfiles.append(testpath + dataset + '_' + type + '_high.npy')
    return testfiles
                

def generate_save_path(prefix, model, dataset, train_type, degree, test_type, test_degree):
    savepath = prefix + '/results/' + model + '/' + dataset + '/'
    savepath = savepath + '/energy_saved/' + train_type + '_' + str(degree) + '_to_' + test_type + '_' + str(test_degree) + '.npy'
    return savepath

def check_testfiles_exist(testfiles):
    for file in testfiles:
        if os.path.exists(file):
            print(file)
            continue
        else:
            # print('error!!! ----' + file)
            return False
    return True

# This function is used to check if the test files exist
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python generate_testfiles.py <dataset>")
        exit(1)
    dataset = sys.argv[1]
    prefix = '../../processed'
    trainpath, labelpath = generate_trainpath_and_label(prefix, dataset, 'pure', 'high')
    testfiles = generate_testfiles(prefix, dataset)
    print("Train Path: ", trainpath)
    print("Label Path: ", labelpath)
    print("Test Files: ", testfiles)
    if not check_testfiles_exist(testfiles):
        print("Test files do not exist!")
        exit(1)

