"""
helper functions for plotting evaluation
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FP_target = 5
neg_txt = 'utils/neg.txt'
labels_path = 'utils/label.csv'
cell_count_path = 'utils/cell_count.csv'

#------------------------------------------------------

def calculate_threshold(csv_dir):
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
  return thresholds

def plot_threshold(csv_dir, out_dir):
  """
  plots threshold for each neg csv
  """
  thresholds = calculate_threshold(csv_dir)
  thresholds_df = pd.DataFrame(thresholds, columns=['file_name', 'threshold'])
  thresholds_df.to_csv(os.path.join(out_dir, 'thresholds.csv'), index=False)
  thresholds_df_sorted = thresholds_df.sort_values(by='threshold')
  
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
  print(f"threshold: {os.path.join(out_dir, 'thresholds_by_file.png')}")

def calculate_fpr_fnr(csv_dir, threshold=0.5):
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

def get_ratio_matrix(csv_dir, threshold_path):
  thresholds_df = pd.read_csv(threshold_path)
  new_labels_df = pd.read_csv(labels_path)
  results = []

  for pos_file in os.listdir(csv_dir):
    if pos_file.endswith('.csv') and pos_file not in neg_txt:
      pos_file_path = os.path.join(csv_dir, pos_file)
      pos_df = pd.read_csv(pos_file_path)
      for i, row in thresholds_df.iterrows():
        threshold = row['threshold']
        ratio = 1 - sum(pos_df['parasite output'] > threshold) / len(pos_df)
          
        name = pos_file[:-4]
        new_label_row = new_labels_df[new_labels_df['name'] == name]
        if not new_label_row.empty:
          name = new_label_row['new label'].values[0]
        else:
          print("name not found, use {}".format(name))

        results.append([name, row['file_name'], ratio])
        
  ratio_df = pd.DataFrame(results, columns=['Pos File', 'Threshold File', 'Ratio'])
  matrix_df = ratio_df.pivot(index='Threshold File', columns='Pos File', values='Ratio')

  return matrix_df

def plot_ratio_matrix(csv_dir, out_dir, text_size=6, tick_size=5):
  """
  Plots the ratio matrix of FNR given FPR and saves it to a file.
  """
  threshold_path = os.path.join(out_dir, 'thresholds.csv')
  out_path = os.path.join(out_dir, 'ratio_matrix.pdf')
  matrix_df = get_ratio_matrix(csv_dir, threshold_path)

  cm = 1/2.54 
  fig, ax = plt.subplots(figsize=(5.8*cm, 4.0*cm))
  c = ax.pcolor(matrix_df, cmap='Blues', vmin=0, vmax=1)

  ax.set_xticks(np.arange(matrix_df.shape[1]), minor=False)
  ax.set_xticklabels([col[:7] for col in matrix_df.columns], minor=False, fontsize=tick_size)
  plt.xticks(rotation=45)

  color_bar = fig.colorbar(c, ax=ax)
  color_bar.ax.tick_params(labelsize=tick_size, width=0.1)
  color_bar.set_label('FNR at FP = 5/µl', fontsize=text_size)

  plt.xlabel('Positive slides', fontsize=text_size)
  plt.ylabel('Negative slides', fontsize=text_size)
  plt.tight_layout()
  plt.tick_params(width=0.1)
  plt.savefig(out_path, dpi=300)
  print(f"ratio matrix: {out_path}")
