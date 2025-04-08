import numpy as np
import random

def noise(df, target_columns, intensity):
  for target_column in target_columns:
    column_data = df[target_column].values
    deviation = np.std(column_data) * intensity
    noise_data = np.random.normal(0, deviation, column_data.shape[0])
    df[target_column] = df[target_column] + noise_data
  return df


def missing(df, target_columns, missing_ratio):
  for target_column in target_columns:
    col_idx = df.columns.get_loc(target_column)
    row_idxs = list(range(df.shape[0]))
    row_idxs = random.sample(row_idxs, int(len(row_idxs) * missing_ratio))
    df.iloc[row_idxs, col_idx] = 0
  return df


def duplicate(df, target_columns, duplicate_ratio, duplicate_length):
  for target_column in target_columns:
    # Each column gets a random duplicate length, and a random duplicate ratio
    col_idx = df.columns.get_loc(target_column)
    row_idxs = list(range(df.shape[0]))
    row_idxs = random.sample(row_idxs, int(len(row_idxs) * duplicate_ratio))
    stuck_length = random.randint(1, duplicate_length)
    for row_idx in row_idxs:
      if row_idx + stuck_length < df.shape[0]:
        df.iloc[row_idx:row_idx + stuck_length, col_idx] = df.iloc[row_idx, col_idx]
  return df



def delay(df, target_columns, delaying_length):
  for target_column in target_columns:
    # Each column gets a random slide step less than the given slide_step threshold
    current_slide = random.randint(1, delaying_length)
    col_idx = df.columns.get_loc(target_column)
    df.iloc[current_slide:, col_idx] = df.iloc[:-current_slide, col_idx].values
  return df


def mismatch(df, target_columns, relative_frequency):
  for target_column in target_columns:
    # Each column gets a random relative frequency
    column_data = df[target_column].values
    step = random.randint(2, relative_frequency)
    sampled_data = np.zeros_like(column_data, dtype=column_data.dtype)
    sampled_data[::step] = column_data[::step]
    # Fill the rest of the sampled data with the last valid value
    for i in range(1, len(sampled_data)):
      if sampled_data[i] == 0:
        sampled_data[i] = sampled_data[i - 1]
    df[target_column] = sampled_data
  return df





