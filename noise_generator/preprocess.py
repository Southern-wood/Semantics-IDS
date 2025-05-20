import numpy as np
import pandas as pd
import random

def noise(df, target_columns, intensity):
  df = df.copy() 
  for target_column in target_columns:
    column_data = df[target_column].values
    deviation = np.std(column_data) * intensity
    noise_data = np.random.normal(0, deviation, column_data.shape[0])
    df[target_column] = df[target_column] + noise_data
  return df


def missing(df, target_columns, missing_ratio):
  df = df.copy()
  for target_column in target_columns:
    col_idx = df.columns.get_loc(target_column)
    row_idxs = list(range(df.shape[0]))
    row_idxs = random.sample(row_idxs, int(len(row_idxs) * missing_ratio))
    df.iloc[row_idxs, col_idx] = 0
  return df


def duplicate(df, target_columns, duplicate_ratio, duplicate_length):
  df = df.copy()
  num_rows = df.shape[0]
  possible_start_indices = np.arange(num_rows)
  num_duplicates = int(num_rows * duplicate_ratio)
  
  for target_column in target_columns:
    # Work on the underlying NumPy array for efficiency
    column_data = df[target_column].values 
    
    # Use np.random.choice for potentially better performance on large arrays
    start_indices = np.random.choice(possible_start_indices, size = num_duplicates, replace=False)

    # Iterate through the selected starting points
    for start_idx in start_indices:
      stuck_length = random.randint(1, duplicate_length) 
      end_idx = min(start_idx + stuck_length, num_rows) 
      value_to_duplicate = column_data[start_idx] 
      if start_idx < end_idx: 
          column_data[start_idx:end_idx] = value_to_duplicate
    df[target_column] = column_data
  return df



def delay(df, target_columns, delaying_length):
  df = df.copy()
  for target_column in target_columns:
    # Each column gets a random slide step less than the given slide_step threshold
    current_slide = random.randint(1, delaying_length)
    col_idx = df.columns.get_loc(target_column)
    df.iloc[current_slide:, col_idx] = df.iloc[:-current_slide, col_idx].values
  return df

