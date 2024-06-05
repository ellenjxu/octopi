"""
helper functions for plotting evaluation
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

FP_target = 5

labels_df = pd.read_csv('utils/label.csv')
def _get_label(file_name): # helper fxn for getting label
  name = file_name[:-4]
  new_label_row = labels_df[labels_df['name'] == name]
  if not new_label_row.empty:
    return new_label_row['new label'].values[0]
  else:
    return name

cell_count_df = pd.read_csv('utils/cell_count.csv')

with open('utils/neg.txt') as f:
  neg_files = f.readlines()
neg_files = [x.strip() for x in neg_files]

#------------------------------------------------------

def calculate_threshold(csv_dir):
  """
  calculates thresholds for LOD 5 FP / 5M RBC
  """
  thresholds = []

  for file_name in neg_files:
    if file_name in os.listdir(csv_dir):
      file_id = file_name[:-4]
      cell_count = cell_count_df.loc[cell_count_df['dataset ID'] == file_id, 'Total Count'].values[0]

      file_path = os.path.join(csv_dir, file_name)
      df = pd.read_csv(file_path)
      df_sorted = df.sort_values(by='parasite output', ascending=False)
      sorted_numbers = df_sorted['parasite output'].to_numpy()

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

def calculate_fpr_fnr(csv_dir, threshold=0.5):
  """
  calculates FPR and FNR for a given threshold.
  """
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

def calculate_fnr(csv_dir, threshold_path):
  """
  fnr based on calculated thresholds
  """
  thresholds_df = pd.read_csv(threshold_path)
  fnr_list = []

  for pos_file in os.listdir(csv_dir):
    if pos_file.endswith('.csv') and pos_file not in neg_files:
      pos_file_path = os.path.join(csv_dir, pos_file)
      pos_df = pd.read_csv(pos_file_path)
      for i, row in thresholds_df.iterrows():
        threshold = row['threshold']
        fnr = 1 - (sum(pos_df['parasite output'] > threshold) / len(pos_df)) # FNR = 1-TP/(TP+FN)
        fnr_list.append([_get_label(pos_file), row['file_name'], fnr])
        
  return fnr_list

def plot_ratio_matrix(csv_dir, out_dir, text_size=6, tick_size=5):
  """
  Plots the ratio matrix of FNR given FPR and saves it to a file.
  """
  threshold_path = os.path.join(out_dir, 'thresholds.csv')
  out_path = os.path.join(out_dir, 'ratio_matrix.pdf')

  fnr_list = calculate_fnr(csv_dir, threshold_path)
  ratio_df = pd.DataFrame(fnr_list, columns=['Pos File', 'Threshold File', 'Ratio'])
  matrix_df = ratio_df.pivot(index='Threshold File', columns='Pos File', values='Ratio')

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

def calculate_confusion_matrix(csv_dir, threshold=0.5):
  """
  calculates TP, FP, TN, FN for each patient.
  """
  confusion_matrix = []

  for file_name in os.listdir(csv_dir):
    file_path = os.path.join(csv_dir, file_name)
    df = pd.read_csv(file_path)

    if file_name in neg_files:
      tp = 0
      fp = np.sum(df['parasite output'] >= threshold)
      tn = np.sum(df['parasite output'] < threshold)
      fn = 0
    else:  # pos
      tp = np.sum(df['parasite output'] >= threshold)
      fp = 0
      tn = 0
      fn = np.sum(df['parasite output'] < threshold)

    name = _get_label(file_name)
    confusion_matrix.append([name, tp, fp, tn, fn])

  return confusion_matrix

def plot_confusion_matrix(csv_dir, out_dir, threshold=0.5):
  """
  Plots the confusion matrix for each patient and saves it to a file.
  """
  confusion_matrix = calculate_confusion_matrix(csv_dir, threshold)
  confusion_df = pd.DataFrame(confusion_matrix, columns=['file_name', 'TP', 'FP', 'TN', 'FN'])
  confusion_df.set_index('file_name', inplace=True)
  confusion_df = confusion_df.div(confusion_df.sum(axis=1), axis=0) # normalize

  plt.figure(figsize=(10, 12))
  sns.heatmap(confusion_df, annot=True, fmt=".4f", cmap="YlGnBu")
  # colors = {'TP': 'Blues', 'FP': 'Reds', 'TN': 'Blues', 'FN': 'Reds'}
    
  # cmap = {
  #   'TP': sns.color_palette("Blues", as_cmap=True),
  #   'FP': sns.color_palette("Reds", as_cmap=True),
  #   'TN': sns.color_palette("Blues", as_cmap=True),
  #   'FN': sns.color_palette("Reds", as_cmap=True)
  # }
    
  # fig, ax = plt.subplots(figsize=(12, 8))
  # for i, column in enumerate(confusion_df.columns):
  #   ax = sns.heatmap(confusion_df[[column]], annot=True, fmt=".2f", cmap=sns.color_palette(colors[column], as_cmap=True),
  #                        cbar=i == 0, linewidths=.5, linecolor='black', yticklabels=i == 0)
  #   ax.collections[0].colorbar.set_label(column)
   
  plt.title('Confusion Matrix for Each Patient')
  plt.xlabel('Metrics')
  plt.ylabel('Patient Label')
  plt.xticks(rotation=45, ha='right')

  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300)
  plt.show()

def calculate_roc(csv_dir):
  """
  calculates joint ROC curve and AUC (batched across patients, so not as accurate)
  """
  y_true = []
  y_scores = []

  for file_name in os.listdir(csv_dir):
    file_path = os.path.join(csv_dir, file_name)
    df = pd.read_csv(file_path)

    if file_name in neg_files:
      y_true.extend([0] * len(df))
    else:
      y_true.extend([1] * len(df))

    y_scores.extend(df['parasite output'].tolist())

  fpr, tpr, thresholds = roc_curve(y_true, y_scores)
  roc_auc = auc(fpr, tpr)

  return fpr, tpr, thresholds, roc_auc

def find_best_threshold(fpr, tpr, thresholds):
  youden_j = tpr - fpr
  best_index = np.argmax(youden_j)
  best_threshold = thresholds[best_index]

  return best_threshold

def plot_roc_curve(csv_dir, out_dir):
  """
  Plots the ROC curve and saves it to a file.
  """
  fpr, tpr, thresholds, roc_auc = calculate_roc(csv_dir)
  best_threshold = find_best_threshold(fpr, tpr, thresholds)

  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.scatter(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], color='red', label=f'Best threshold = {best_threshold:.2f}')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc="lower right")
  plt.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=300)
  plt.show()

  return best_threshold