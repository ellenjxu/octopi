import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

# Define the path to the folder containing .npy files
folder_path = 'out/resnet18/h7_v4_newsbc/csv_whole'
save_path =   'out/resnet18/h7_v4_newsbc/count'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    if len(sys.argv) > 2:
        save_path = sys.argv[2]

results = []
result_save_indicator = False

#for thresh in tqdm(np.arange(0.01, 1.0, 0.01)):
for thresh in [0.5]:

    # check if this threshold has been calculated
    #if os.path.exists(os.path.join(save_path, 'all_ds_prediction_counts_{:.3f}.csv'.format(thresh))):
    #    continue

    # print("List of datasets to export and infer have been created.")
    #with open('utils/list.txt','r') as dataset_file:
    #    datasets_to_infer = [line.strip().split(".npy")[0] for line in dataset_file.readlines()]

    ## add segmentation stats
    df2 = pd.read_csv('utils/cell_count.csv')
    # check column "ML", if there is and "train" in it, remove it
    if 'ML' in df2.columns and 'Train_neg' in df2['ML'].values: # this is because Train_neg is not in the current folder
        df2 = df2[df2['ML'] != 'Train_neg']

    datasets_to_infer = df2['dataset ID'].tolist()

    data = {
        "dataset ID": datasets_to_infer,
        "predicted positive": [-1] * len(datasets_to_infer),
        "predicted negative": [-1] * len(datasets_to_infer),
        "predicted unsure": [-1] * len(datasets_to_infer)
    }
    all_dataset_prediction_counts = pd.DataFrame(data)

    for dataset_id_0 in datasets_to_infer:
        dataset_id = folder_path + '/' + dataset_id_0
        #print(dataset_id_0)

        # USER PARAMETERS (optional)
        unsure_ignored = True 

        # intermediate / output paths
        path_csv_annotations_and_predictions = dataset_id + '.csv'

        # Read the CSV file into a DataFrame
        df = pd.read_csv(path_csv_annotations_and_predictions)

        # Calculate pred_pos, pred_neg, and pred_unsure based on your conditions
        pred_pos = len(df[df['parasite output'] >= thresh])
        pred_neg = len((df['parasite output'] < thresh))
        pred_unsure = len(df) - pred_pos - pred_neg
        all_dataset_prediction_counts.loc[all_dataset_prediction_counts['dataset ID'] == dataset_id_0, ['predicted positive', 'predicted negative', 'predicted unsure']] = [pred_pos, pred_neg, pred_unsure]

    # Renaming the 'Dataset ID' column in df2 to match the 'dataset ID' column in df1 for consistency
    #df2 = df2.rename(columns={"Dataset ID": "dataset ID"})
    # Merging the datasets on 'dataset ID'
    merged_df = pd.merge(df2, all_dataset_prediction_counts, on="dataset ID")
    #print(merged_df.columns)
    # print the columns of df2 and merged_df and all_dataset_prediction_counts
    # Calculating the number of positives per (Total Count / 5e6)
    merged_df['Positives per 5M RBC'] = merged_df['predicted positive'] / (merged_df['Total Count'] / 5e6)

    # save
    # if the directory does not exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    merged_df.to_csv(os.path.join(save_path, 'all_ds_prediction_counts_{:.3f}.csv'.format(thresh)))
