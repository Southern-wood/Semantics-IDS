import os
import pandas as pd

def load_SWaT(dataset_folder):
  dataset_folder = os.path.join(dataset_folder, 'SWaT')
  attack_path = os.path.join(dataset_folder, "SWaT_Dataset_Attack_v0.csv")
  normal_path = os.path.join(dataset_folder, "SWaT_Dataset_Normal_v1.csv")
  info_path = os.path.join(dataset_folder, "List_of_attacks_Final.csv")

  print("SWaT dataset loaded.")

  attack_pd = pd.read_csv(attack_path)
  normal_pd = pd.read_csv(normal_path, skiprows=1)

  column_names = [
    'Timestamp', 'FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201',
    'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203',
    'P204', 'P205', 'P206', 'DPIT301', 'FIT301', 'LIT301', 'MV301',
    'MV302', 'MV303', 'MV304', 'P301', 'P302', 'AIT401', 'AIT402',
    'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404', 'UV401', 'AIT501',
    'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504',
    'P501', 'P502', 'PIT501', 'PIT502', 'PIT503', 'FIT601', 'P601', 'P602',
    'P603', 'Normal/Attack'
  ]

  redunant_columns = ['P202', 'P404', 'P502', 'P601', 'P603']

  for current_name in attack_pd.columns:
    if " " in current_name:
      new_name = current_name.replace(" ", "")
      attack_pd = attack_pd.rename(columns={current_name: new_name})

  for current_name in normal_pd.columns:
    if " " in current_name:
      new_name = current_name.replace(" ", "")
      normal_pd = normal_pd.rename(columns={current_name: new_name})

  attack_pd['Timestamp'] = attack_pd['Timestamp'].str.strip()
  normal_pd['Timestamp'] = normal_pd['Timestamp'].str.strip()
  attack_pd['Timestamp'] = pd.to_datetime(attack_pd['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
  normal_pd['Timestamp'] = pd.to_datetime(normal_pd['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')

  attack_pd.dropna(how='any', inplace=True)
  normal_pd.dropna(how='any', inplace=True)
  attack_pd.fillna(0, inplace=True)
  normal_pd.fillna(0, inplace=True)

  # Remove redunant columns and identify categorical columns
  attack_pd = attack_pd.drop(redunant_columns, axis=1)
  normal_pd = normal_pd.drop(redunant_columns, axis=1)
  print("Redunant Columns removed.")

  # Load labels
  attack_info = pd.read_csv(info_path, index_col=['Attack'])
  attack_info['Start Time'] = pd.to_datetime(attack_info['Start Time'], dayfirst=True)
  attack_info['End Time'] = pd.to_datetime(attack_info['End Time'], dayfirst=True)

  labels = attack_pd.copy(deep=True)
  for i in attack_pd.columns.tolist()[1:-1]:
    labels[i] = 0
  for index, row in attack_info.iterrows():
    matched = row['Attack Point'].lstrip("\"").rstrip("\"").replace("-", "").replace(" ", "").split(',')
    st, et = str(row['Start Time']), str(row['End Time'])
    for one_match in matched:
      if one_match in column_names:
        labels.loc[(labels['Timestamp'] >= st) & (labels['Timestamp'] <= et), one_match] = 1

  #Attack 4 one cate column single matched point not exist, add manually and choose the first column as affected column
  attack_4_start,attack_4_end = "2015-12-28 11:47:39","2015-12-28 11:54:08"
  labels.loc[(labels['Timestamp'] >= attack_4_start) & (labels['Timestamp'] <= attack_4_end), 'FIT101'] = 1

  index = attack_pd.columns[1:-1]
  attack_pd, normal_pd, labels = attack_pd[index], normal_pd[index], labels[index]

  categorical_column = [
      idx for idx, column in enumerate(attack_pd.columns)
      if all(isinstance(value, int) and value == int(value) for value in attack_pd[column])
  ]
  print("SWaT dataset prepared.")

  return normal_pd, attack_pd, labels, categorical_column


def load_WADI(dataset_folder):
  dataset_folder = os.path.join(dataset_folder, 'WADI')
  train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=4)
  test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
  ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
  print('WADI dataset loaded.')

  train.dropna(how='all', inplace=True)
  test.dropna(how='all', inplace=True)
  train.fillna(0, inplace=True)
  test.fillna(0, inplace=True)

  test['Time'] = test['Time'].astype(str)
  test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'], format='%m/%d/%Y %I:%M:%S.%f %p')

  # Check the train and test original column names are same, remove the redunant columns
  test_column_names = test.columns.to_list()
  for index, column_name in enumerate(test_column_names):
    if "\\" in column_name:
      name_split = column_name.split("\\")
      test = test.rename(columns={column_name: name_split[-1]})

  train_column_names = train.columns.to_list()
  for index, column_name in enumerate(train_column_names):
    if "\\" in column_name:
      name_split = column_name.split("\\")
      train = train.rename(columns={column_name: name_split[-1]})

  train_column_names = train.columns.to_list()
  test_column_names = test.columns.to_list()

  nan_columns = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']
  remove_columns = [
    '1_LS_001_AL', '1_LS_002_AL', '1_P_002_STATUS', '1_P_004_STATUS', '2_MV_001_STATUS',
    '2_MV_002_STATUS', '2_MV_004_STATUS', '2_MV_005_STATUS', '2_MV_009_STATUS', '2_P_004_STATUS',
    '2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', '2_SV_501_STATUS',
    '2_SV_601_STATUS', '3_LS_001_AL', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS',
    '3_P_001_STATUS', '3_P_002_STATUS', '3_P_003_STATUS', '3_P_004_STATUS', 'PLANT_START_STOP_LOG'
  ]
  redunant_columns = nan_columns + remove_columns

  train = train.drop(redunant_columns, axis=1)
  test = test.drop(redunant_columns, axis=1)
  print("Redunant Columns removed.")

  # Load labels
  labels = test.copy(deep=True)
  for i in test.columns.tolist()[3:]:
    labels[i] = 0
  for i in ['Start Time', 'End Time']:
    ls[i] = ls[i].astype(str)
    ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
  for index, row in ls.iterrows():
    to_match = row['Affected'].split(', ')
    matched = []
    for i in test.columns.tolist()[3:]:
      for tm in to_match:
        if tm in i:
          matched.append(i)
          break
    st, et = str(row['Start Time']), str(row['End Time'])
    labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1

  index = train.columns[3:]
  train, test, labels = train[index], test[index], labels[index]
  categorical_column = [
    idx for idx, column in enumerate(train.columns)
    if all(isinstance(value, int) and value == int(value) for value in train[column])
  ]
  print("WADI dataset prepared.")
  return train, test, labels, categorical_column


def load_HAI(data_folder):
  dataset_folder = os.path.join(data_folder, 'HAI')
  train = pd.read_csv(os.path.join(dataset_folder, 'train2.csv'), sep=';')
  test = pd.read_csv(os.path.join(dataset_folder, 'test2.csv'), sep=';')
  print("HAI dataset loaded.")

  train.dropna(how='all', inplace=True)
  test.dropna(how='all', inplace=True)
  train.fillna(0, inplace=True)
  test.fillna(0, inplace=True)

  redunant_columns = []
  for column in train.columns[:-3]:
    if train[column].nunique() == 1 or test[column].nunique() == 1:
      redunant_columns.append(column)

  train = train.drop(redunant_columns, axis=1)
  test = test.drop(redunant_columns, axis=1)
  print("Redunant Columns removed.")

  # Load labels
  labels = test.copy(deep=True)
  for i in test.columns.tolist()[1:-3]:
    labels[i] = 0

  attacked_processes = ['P1', 'P2', 'P3']
  attack_rows = test[test['attack'] == 1]

  for process in attacked_processes:
    process_columns = [column for column in test.columns if process in column]
    affected_indices = attack_rows[attack_rows['attack_' + process] == 1].index

    labels.loc[affected_indices, process_columns] = 1

  index = train.columns[1:-4]
  train, test, labels = train[index], test[index], labels[index]
  categorical_column = [
    idx for idx, column in enumerate(train.columns)
    if all(isinstance(value, int) and value == int(value) for value in train[column])
  ]
  print("HAI dataset prepared.")
  return train, test, labels, categorical_column

# Main function for testing only
if __name__ == "__main__":
  dataset_folder = 'dataset'
  load_SWaT(dataset_folder)
  load_WADI(dataset_folder)
  load_HAI(dataset_folder)