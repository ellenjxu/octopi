"""
helper functions for plotting evaluation
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FP_target = 5

def calculate_threshold(csv_dir, neg_txt, cell_count_path, out_dir):
  """
  calculates thresholds for LOD 5 FP / 5M RBC
  """
  with open(neg_txt) as f:
    neg_files = f.readlines()
  neg_files = [x.strip() for x in neg_files]

  thresholds = []
  cell_count_df = pd.read_csv(cell_count_path)

  for file_name in neg_files:
    if file_name in os.listdir(csv_dir):
      file_id = file_name[:-4]
      cell_count = cell_count_df.loc[cell_count_df['dataset ID'] == file_id, 'Total Count'].values[0]

      file_path = os.path.join(csv_dir, file_name)
      df = pd.read_csv(file_path)
      df_sorted = df.sort_values(by='parasite output', ascending=False)
      parasite_output_array = df_sorted['parasite output'].to_numpy()
      sorted_numbers = np.sort(parasite_output_array)

      for i, num in enumerate(sorted_numbers):
        normalized_count = (i + 1) / cell_count
        if normalized_count > FP_target / 5e6:
          threshold = num
          break
      
      thresholds.append([file_name, threshold])
      
  thresholds_df = pd.DataFrame(thresholds, columns=['file_name', 'threshold'])
  thresholds_df.to_csv(os.path.join(out_dir, 'thresholds.csv'), index=False)
  thresholds_df_sorted = thresholds_df.sort_values(by='threshold')

  return thresholds_df_sorted

def plot_threshold(thresholds_df_sorted, out_dir):
  """
  plots threshold for each neg csv
  """
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  
  plt.figure(figsize=(10,8))
  x_labels = range(1, len(thresholds_df_sorted) + 1)
  plt.bar(x_labels, thresholds_df_sorted['threshold'])
  plt.axhline(y=0.98, color='r', linestyle='--',label='0.98')

  plt.title('Thresholds by File')
  plt.xlabel('File Name')
  plt.ylabel('Threshold')
  plt.xticks(rotation=45, ha='right')

  plt.tight_layout() 
  plt.savefig(os.path.join(out_dir, 'thresholds_by_file.png'), dpi=300)

def calculate_fpr_fnr(csv_dir, neg_txt, threshold=0.5):
  """
  calculates FPR and FNR for a given threshold.
  """
  with open(neg_txt) as f:
    neg_files = [x.strip() for x in f.readlines()]

  fpr_list, fnr_list = [], []

  for file_name in os.listdir(csv_dir):
    file_path = os.path.join(csv_dir, file_name)
    df = pd.read_csv(file_path)
    total = len(df)

    if file_name in neg_files:
      fpr = np.sum(df['parasite output'] >= threshold) / total
      fpr_list.append(fpr)
    else: # pos
      fnr = np.sum(df['parasite output'] < threshold) / total
      fnr_list.append(fnr)

  return fpr_list, fnr_list
