"""
helper functions for plotting evaluation
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay

cm = 1/2.54 

# Global variable to control figure size
MANUSCRIPT_MODE = True

# Helper function to set figure properties based on mode
def set_figure_properties(fig, ax, fontsize, linewidth):
    if MANUSCRIPT_MODE:
        ax.tick_params(width=linewidth, length=2*linewidth)
        for spine in ax.spines.values():
            spine.set_linewidth(linewidth)
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['axes.linewidth'] = linewidth
    else:
        # Default settings for better visibility
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.5
        ax.tick_params(width=1.5, length=6)
        # tight layout

labels_df = pd.read_csv('utils/label.csv')
def _get_label(file_name): # helper fxn for getting label
  name = file_name[:-4]
  new_label_row = labels_df[labels_df['name'] == name]
  if not new_label_row.empty:
    return new_label_row['new label'].values[0]
  else:
    return name
  
def _get_name(label): # helper fxn for getting name
  new_label_row = labels_df[labels_df['new label'] == label]
  if not new_label_row.empty:
    return new_label_row['name'].values[0]
  else:
    return label

cell_count_df = pd.read_csv('utils/cell_count.csv')

with open('utils/neg.txt') as f:
  neg_files = f.readlines()
neg_files = [x.strip() for x in neg_files]

#------------------------------------------------------

def calculate_threshold(csv_dir,FP_target):
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

def plot_threshold(csv_dir, out_dir,FP_target=5):
  """
  plots threshold for each neg csv
  """

  if MANUSCRIPT_MODE:
    figsize = (5.8*cm, 4.0*cm)  # in cm
    fontsize = 6
    linewidth = 0.1
  else:
    figsize = (12, 8)  # in inches
    fontsize = 12
    linewidth = 1.5

  fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
  set_figure_properties(fig, ax, fontsize, linewidth)

  thresholds = calculate_threshold(csv_dir,FP_target)
  thresholds_df = pd.DataFrame(thresholds, columns=['file_name', 'threshold'])
  # use _get_label to get the label for each file
  thresholds_df['label'] = thresholds_df['file_name'].apply(_get_label)
  thresholds_df.to_csv(os.path.join(out_dir, 'thresholds.csv'), index=False)

  df = thresholds_df
  
  ax.bar(range(len(df['threshold'])), df['threshold'])
  for i, threshold, label in zip(range(len(df)), df['threshold'], df['label']):
      ax.text(i, threshold, f"{label}", ha='center', va='bottom', fontsize=fontsize)

  ninefive_pertile = df['threshold'].quantile(0.95)
  line = ax.axhline(y=ninefive_pertile, color='r', linestyle='--', label=f'95% Pertile: {ninefive_pertile:.2f}')
  ax.legend(handles=[line], fontsize=fontsize)

  ax.set_title('Thresholds by File', fontsize=fontsize+2)
  ax.set_xlabel('File Name', fontsize=fontsize)
  ax.set_ylabel('Threshold', fontsize=fontsize)

  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, 'thresholds_by_file.pdf'), dpi=300)

  return ninefive_pertile

def calculate_fpr_fnr(csv_dir, threshold=0.5):
  """
  calculates FPR, FNR and FPs per 5M RBC for a given threshold.
  """
  fpr_list, fnr_list, fps = [], [] ,[]

  for file_name in os.listdir(csv_dir):
    file_path = os.path.join(csv_dir, file_name)
    df = pd.read_csv(file_path)
    total = len(df)

    if file_name in neg_files:
      fpr = np.sum(df['parasite output'] >= threshold) / total
      fpr_list.append(fpr)

      file_id = file_name.split('.csv')[0]
      cell_count = cell_count_df.loc[cell_count_df['dataset ID'] == file_id, 'Total Count'].values[0]
      fps.append(np.sum(df['parasite output'] >= threshold) / cell_count * 5e6)
    
    else: # pos
      fnr = np.sum(df['parasite output'] < threshold) / total
      fnr_list.append(fnr)

  return fpr_list, fnr_list, fps

def plot_fp_fnr(csv_dir, out_dir, thr_start=0.05):
  '''
  Plot the FNR and FPs per 5m RBC in one graph
  '''

  if MANUSCRIPT_MODE:
    figsize = (7, 4)  # in cm
    fontsize = 6 # for the labels (text fontsize)
    ticksize = 5 # for the tick's font size
    linewidth = 1 # the plot line width
    markersize = 10 # the size of the markers
    axislinewidth = 0.5 # the axis line width
  else:
    figsize = (30, 20) 
    fontsize = 12
    linewidth = 1.5
  
  fig, ax = plt.subplots(figsize=(figsize[0]/2.54, figsize[1]/2.54))
  # plot two y-axis, left for FPs per 5M RBC, right for FNR
  ax2 = ax.twinx()
  set_figure_properties(fig, ax, fontsize, axislinewidth)
  set_figure_properties(fig, ax2, fontsize, axislinewidth)
  
  blue = '#3182BD'
  red = '#F03B20'

  # store all the df in the buffer
  dfs = []
  for file_name in os.listdir(csv_dir):
    file_path = os.path.join(csv_dir, file_name)
    df = pd.read_csv(file_path)
    total = len(df)
    dfs.append(df)
    
  #thrs = np.linspace(thr_start, 1.01, num=50)
  thrs = np.arange(thr_start, 1.01, 0.01)
  thrs[-1] = 1+1e-6
  
  fps_slides = []
  fnr_slides = []
  fpr_slides = []
  for thr in tqdm(thrs):
    #fpr_list, _, fps = calculate_fpr_fnr(csv_dir, thr)

    fnr_list, fps ,fpr_list = [], [] , [] 
    for file_name, df in zip(os.listdir(csv_dir), dfs):
      total = len(df)
      if file_name in neg_files:
        file_id = file_name.split('.csv')[0]
        cell_count = cell_count_df.loc[cell_count_df['dataset ID'] == file_id, 'Total Count'].values[0]
        #print(cell_count)
        fps.append(np.sum(df['parasite output'] >= thr) / cell_count * 5e6)
        fpr = np.sum(df['parasite output'] >= thr) / total
        fpr_list.append(fpr)
        
      else:
        #print(np.sum(df['parasite output'] < thr))
        fnr = np.sum(df['parasite output'] < thr) /total
        fnr_list.append(fnr)

    fpr_slides.append((fpr_list))
    fnr_slides.append((fnr_list))
    fps_slides.append((fps))

  # reshape the data
  fnr_slides = np.array(fnr_slides).transpose()
  fps_slides = np.array(fps_slides).transpose()
  fpr_slides = np.array(fpr_slides).transpose()
  
  
  # save the data as csv fith slide name and threshold
  fnr_df = pd.DataFrame(fnr_slides, columns=[f'thr_{thr}' for thr in thrs])
  fnr_df['file_name'] = [file_name for file_name in os.listdir(csv_dir) if file_name not in neg_files]
  fnr_df.to_csv(os.path.join(out_dir, 'fnr.csv'), index=False)

  fp_df = pd.DataFrame(fps_slides, columns=[f'thr_{thr}' for thr in thrs])
  fp_df['file_name'] = [file_name for file_name in os.listdir(csv_dir) if file_name in neg_files]
  fp_df.to_csv(os.path.join(out_dir, 'fp.csv'), index=False)

  fpr_df = pd.DataFrame(fpr_slides, columns=[f'thr_{thr}' for thr in thrs])
  fpr_df['file_name'] = [file_name for file_name in os.listdir(csv_dir) if file_name  in neg_files]
  fpr_df.to_csv(os.path.join(out_dir, 'fpr.csv'), index=False)

  for i in range(fps_slides.shape[0]):
    ax.plot(thrs, fps_slides[i], label=f'FPs {i}', alpha=0.5, color=red,linewidth=linewidth)

  for i in range(fnr_slides.shape[0]):
    ax2.plot(thrs, fnr_slides[i], label=f'FNR {i}', alpha=0.5, color=blue,linewidth=linewidth)

  # compute an average fnr 
  fnr_avg = np.mean(fnr_slides, axis=0)
  # plot with pink with clear markers
  #line = ax2.plot(thrs, fnr_avg, label='FNR Avg', color='orange', marker='o', linewidth=linewidth)
  # show the exact number at fnr_avg[0] in the plot
  #ax2.text(thrs[0], fnr_avg[0], f'{fnr_avg[0]:.2f}', ha='right', va='bottom', fontsize=5)
  # shows the legend only for the average
  #ax2.legend(handles=line)


  ax.set_xlabel('Threshold', fontsize=fontsize)
  ax.set_ylabel('False positive count / µL', color=red, fontsize=fontsize)
  ax.tick_params(axis='y', labelcolor=red)
  ax.set_ylim(0, 15)
  ax2.set_ylabel('Per spot false negative rate', color=blue, fontsize=fontsize)
  ax2.tick_params(axis='y', labelcolor=blue)
  ax2.set_ylim(0, 1)
  
  plt.tight_layout()
  fig.savefig(os.path.join(out_dir, f'fp_fnr.pdf'), dpi=300)

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

def plot_ratio_matrix(csv_dir, out_dir,FP_target=5):
  """
  Plots the ratio matrix of FNR given FPR and saves it to a file.
  """
  if MANUSCRIPT_MODE:
      figsize = (7, 4)  # in cm
      fontsize = 6 # for the labels (text fontsize)
      ticksize = 5 # for the tick's font size
      linewidth = 1 # the plot line width
      markersize = 10 # the size of the markers
      axislinewidth = 0.5 # the axis line width
  else:
      figsize = (12, 10)  # in inches
      fontsize = 12
      linewidth = 1.5
      ticksize=8
  
  fig, ax = plt.subplots(figsize=(figsize[0]/2.54, figsize[1]/2.54))
  set_figure_properties(fig, ax, fontsize, linewidth)

  threshold_path = os.path.join(out_dir, 'thresholds.csv')

  fnr_list = calculate_fnr(csv_dir, threshold_path)
  ratio_df = pd.DataFrame(fnr_list, columns=['Pos File', 'Threshold File', 'Ratio'])
  matrix_df = ratio_df.pivot(index='Threshold File', columns='Pos File', values='Ratio')

  c = ax.pcolor(matrix_df, cmap='Blues', vmin=0, vmax=1)

  ax.set_xticks(np.arange(matrix_df.shape[1]), minor=False)
  ax.set_xticklabels([col[:7] for col in matrix_df.columns], minor=False, fontsize=ticksize)
  plt.xticks(rotation=45)

  if not MANUSCRIPT_MODE:
    ax.set_yticks(np.arange(matrix_df.shape[0]), minor=False)
    ax.set_yticklabels([_get_label(file_name) for file_name in matrix_df.index], minor=False, fontsize=ticksize)
  else:
     # show every 10th label, add "H" before the number
    ax.set_yticks(np.arange(matrix_df.shape[0])[::10], minor=False)
    # just show the number not the name
    ax.set_yticklabels([f"H{int(i*10)}" for i in range(len(matrix_df.index[::10]))], minor=False, fontsize=ticksize)

  ax.tick_params(axis='both', which='major', labelsize=ticksize, width=axislinewidth)
  color_bar = fig.colorbar(c, ax=ax)
  color_bar.ax.tick_params(labelsize=ticksize, width=axislinewidth)
  color_bar.set_label('FNR at FP = {}/µl'.format(FP_target), fontsize=fontsize)

  for spine in ax.spines.values():
    spine.set_linewidth(axislinewidth)

  # set the axislinewidth of the colorbar
  color_bar.outline.set_linewidth(axislinewidth)

  plt.xlabel('Positive slides', fontsize=fontsize)
  plt.ylabel('Negative slides', fontsize=fontsize)
  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, f'ratio_matrix.pdf'), dpi=300)

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

def plot_roc_curve(csv_dir, out_dir,fpr_end = 0.0001,fpr_cutoff = 0.5,tpr_cutoff = 0.95):
  """
  Plots the ROC curve and saves it to a file.
  """
  if MANUSCRIPT_MODE:
    figsize = (5.8, 4.0)  # in cm
    fontsize = 6
    linewidth = 0.3
    markersize = 1 # the size of the markers
    axislinewidth = 0.5 # the axis line width
  else:
    figsize = (20, 15)  # in inches
    fontsize = 12
    linewidth = 1.5
  
  fig, ax = plt.subplots(figsize=(figsize[0]/2.54, figsize[1]/2.54))
  set_figure_properties(fig, ax, fontsize, linewidth)

  fpr, tpr, thresholds, roc_auc = calculate_roc(csv_dir)

  ax.plot(fpr, tpr, color='darkorange', lw=linewidth*3)
  ax.plot([0, 1], [0, 1], color='navy', lw=linewidth*2, linestyle='--')

  ax.axvline(x=fpr_cutoff, color='r', linestyle='--', label=f'FPR: {fpr_cutoff:.5f}',linewidth=linewidth)
  ax.scatter(fpr_cutoff, tpr_cutoff, color='r', s=markersize)
  ax.text(fpr_cutoff, tpr_cutoff, f" TPR: ({tpr_cutoff:.4f})", ha='left', va='top', fontsize=fontsize)

  ax.set_xlim([0.0, fpr_end])
  # use e notation for the x-axis
  ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
  ax.set_ylim([0.0, 1.0])
  ax.set_xlabel('False Positive Rate', fontsize=fontsize)
  ax.set_ylabel('True Positive Rate', fontsize=fontsize)
  #ax.set_title('Receiver Operating Characteristic', fontsize=fontsize+2)
  ax.legend(loc="lower right", fontsize=fontsize-2)
  
  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, 'roc_curve_zoomed.pdf'), dpi=300)

  # plot the full ROC curve
  fig, ax = plt.subplots(figsize=(figsize[0]/2.54, figsize[1]/2.54))
  set_figure_properties(fig, ax, fontsize, linewidth)

  ax.plot(fpr, tpr, color='darkorange', lw=linewidth*6, label=f'ROC curve (area = {roc_auc:.5f})',zorder=2)
  ax.plot([0, 1], [0, 1], color='navy', lw=linewidth*4, linestyle='--')

  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.0])
  ax.set_xlabel('False Positive Rate', fontsize=fontsize)
  ax.set_ylabel('True Positive Rate', fontsize=fontsize)
  #ax.set_title('Receiver Operating Characteristic', fontsize=fontsize+2)
  ax.legend(loc="lower right", fontsize=fontsize-2)

  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, 'roc_curve.pdf'), dpi=300)

def merge(path1, path2, model1, model2, ver1, ver2):
    if MANUSCRIPT_MODE:
        figsize = (17.5, 5)  # in cm
        fontsize = 6 # for the labels (text fontsize)
        ticksize = 5 # for the tick's font size
        linewidth = 1 # the plot line width
        markersize = 10 # the size of the markers
        axislinewidth = 0.5 # the axis line width
    else:
        figsize = (30, 20) 
        fontsize = 12
        linewidth = 1.5
    
    fig, ax = plt.subplots(figsize=(figsize[0]/2.54, figsize[1]/2.54))
    set_figure_properties(fig, ax, fontsize, axislinewidth)

    dir1 = f"{path1}/{model1}/{ver1}/csv"
    dir2 = f"{path2}/{model2}/{ver2}/csv"
    save_dir = f"{path1}/{model1}/{ver1}_{ver2}/csv"

    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        print(f"Found {len(files1)} files in {dir1}")
        print(f"Found {len(files2)} files in {dir2}")
        print(f"Saving to {save_dir}")

        for file in tqdm(files1):
            df1 = pd.read_csv(f"{dir1}/{file}")
            df2 = pd.read_csv(f"{dir2}/{file}")

            indices = df1["parasite output"] > df2["parasite output"]
            df1.loc[indices] = df2.loc[indices]

            df1.to_csv(f"{save_dir}/{file}", index=False)

    return 

    FP_target = 5
    thresholds1 = calculate_threshold(dir1, FP_target)
    thresholds2 = calculate_threshold(dir2, FP_target)
    threshold_merged = calculate_threshold(save_dir, FP_target)

    thresholds1_df = pd.DataFrame(thresholds1, columns=['file_name', 'threshold'])
    thresholds2_df = pd.DataFrame(thresholds2, columns=['file_name', 'threshold'])
    thresholds_merged_df = pd.DataFrame(threshold_merged, columns=['file_name', 'threshold'])

    thresholds1_df['label'] = thresholds1_df['file_name'].apply(_get_label)
    thresholds2_df['label'] = thresholds2_df['file_name'].apply(_get_label)
    thresholds_merged_df['label'] = thresholds_merged_df['file_name'].apply(_get_label)

    assert (thresholds1_df['label'] == thresholds2_df['label']).all()
    assert (thresholds1_df['label'] == thresholds_merged_df['label']).all()
    assert (thresholds2_df['label'] == thresholds_merged_df['label']).all()

    df1 = thresholds1_df
    df2 = thresholds2_df
    df_merged = thresholds_merged_df

    x = np.arange(len(df1))

    plot_dir = os.path.join(f"{path1}/{model1}/{ver1}_{ver2}/plots")

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    figure = "scatter"  # You can change this to "bar" or "scatter" if needed

    if figure == "bar":
        rects1 = ax.bar(x, df1['threshold'], label=ver1, alpha=0.3, color='b')
        rects2 = ax.bar(x, df2['threshold'], label=ver2, alpha=0.3, color='r')

        ax.set_xticks(x)
        for i, thr_merged, label in zip(range(len(df1)), df_merged['threshold'], df1['label']):
            ax.text(i, thr_merged, f"{label}", ha='center', va='bottom', fontsize=fontsize-2)

        ninefive_pertile = df_merged['threshold'].quantile(0.95)
        line = ax.axhline(y=ninefive_pertile, color='r', linestyle='--', label=f'95% Pertile: {ninefive_pertile:.2f}')
        fig.legend(handles=[line, rects1, rects2], fontsize=fontsize-2)

        ax.set_title('Thresholds by File', fontsize=fontsize+2)
        ax.set_xlim(0, len(x))
        ax.set_xlabel('File Name', fontsize=fontsize)
        ax.set_ylabel('Threshold', fontsize=fontsize)

    elif figure == "scatter":
        scatter1 = ax.scatter(x, df1['threshold'], label='Model 1', color='#1f77b4', alpha=1, s=markersize, zorder=2)
        scatter2 = ax.scatter(x, df2['threshold'], label='Model 2', color='#ff7f0e', alpha=1, s=markersize, zorder=2)
        scatter3 = ax.scatter(x, df_merged['threshold'], label='Ensemble model', color='#2ca02c', alpha=1, s=markersize, marker='*', zorder=3)

        for i in range(len(x)):
            ax.plot([x[i], x[i]], [df1['threshold'][i], df2['threshold'][i]], color='gray', linestyle='--', alpha=0.7, linewidth=linewidth)
            ax.plot([x[i], x[i]], [df_merged['threshold'][i], min(df1['threshold'][i], df2['threshold'][i])], color='#2ca02c', linestyle='--', alpha=1, linewidth=linewidth)

        ax.set_xlabel('Dataset ID', fontsize=fontsize)
        ax.set_ylabel('Threshold', fontsize=fontsize)

        # set x-ticks, add a "H" before the number to indicate that it is a healthy slide, tick every 20 slides
        ax.set_xlim(-1, len(x))
        ax.set_xticks(x[::20])
        ax.set_xticklabels([f"H{x}" for x in x[::20]], fontsize=ticksize)


        # set tick size
        ax.tick_params(axis='both', which='major', labelsize=ticksize)

        ax.legend(fontsize=fontsize-2)

        

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'merged_{figure}_plot.pdf'), dpi=300)
    plt.close(fig)

    return

# Usage example:
# merge(path1, path2, model1, model2, ver1, ver2, manuscript_mode=True)

def numpy_array_to_image_string(frame):
    frame = frame.transpose(1, 2, 0)
    img_fluorescence = frame[:, :, [2, 1, 0]]
    img_dpc = frame[:, :, 3]
    img_dpc = np.dstack([img_dpc, img_dpc, img_dpc])
    img_overlay = 0.64 * img_fluorescence + 0.36 * img_dpc

    return img_overlay


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
      fp = np.sum(df['parasite output'] > threshold)
      tn = np.sum(df['parasite output'] <= threshold)
      fn = 0
    else:  # pos
      tp = np.sum(df['parasite output'] > threshold)
      fp = 0
      tn = 0
      fn = np.sum(df['parasite output'] <= threshold)

    name = _get_label(file_name)
    confusion_matrix.append([name, tp, fp, tn, fn])

  return confusion_matrix

def plot_confusion_matrix(csv_dir, out_dir, threshold=0.5):
    """
    Plots the confusion matrix for the entire dataset and saves it to a file.
    """
    if MANUSCRIPT_MODE:
        figsize = (5.8, 5.5)  # in cm
        fontsize = 6
        ticksize = 5
        linewidth = 1 # the plot line width
        markersize = 10 # the size of the markers
        axislinewidth = 0.5 # the axis line width
    else:
        figsize = (10, 8)  # in inches
        fontsize = 12
        linewidth = 1.5
    
    fig, ax = plt.subplots(figsize=(figsize[0]/2.54, figsize[1]/2.54))

    confusion_matrix = calculate_confusion_matrix(csv_dir, threshold)
    confusion_df = pd.DataFrame(confusion_matrix, columns=['Slide', 'TP', 'FP', 'TN', 'FN'])
    
    confusion_df = confusion_df[['TP', 'FP', 'FN','TN']].sum()
  
    TPR = confusion_df['TP'] / (confusion_df['TP'] + confusion_df['FN'])
    FPR = confusion_df['FP'] / (confusion_df['FP'] + confusion_df['TN'])
    TNR = confusion_df['TN'] / (confusion_df['TN'] + confusion_df['FP'])
    FNR = confusion_df['FN'] / (confusion_df['TP'] + confusion_df['FN'])

    cm = np.array([[TPR, FNR], [FPR, TNR]])

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    cbar = fig.colorbar(im, ax=ax)
    
    # colorbar ticks line width
    cbar.outline.set_linewidth(axislinewidth)
    cbar.ax.tick_params(labelsize=ticksize, width=axislinewidth)
        
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['P', 'N'], 
           yticklabels=['P', 'N'],
           ylabel='True',
           xlabel='Predicted')

    #ax.set_title('Confusion Matrix', fontsize=fontsize+2)
    ax.set_xlabel('Predicted Label', fontsize=fontsize)
    ax.set_ylabel('True Label', fontsize=fontsize)

    fmt = '.3f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]*100:{fmt}}%',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=fontsize-2)
    set_figure_properties(fig, ax, fontsize, linewidth)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.pdf'), dpi=300)
    plt.close()

    return TPR, FPR, TNR, FNR