import pandas as pd

# load csv
df = pd.read_csv('out/resnet18/h7_v4_newsbc/count/all_ds_prediction_counts_0.500.csv')

df = df[df['ML'].isin(['Test_neg', 'Test_pos_spot', 'Test_pos_patient'])]
# Map ML values to positive/negative
df['ML'] = df['ML'].map({'Test_neg': 'negative', 'Test_pos_spot': 'positive', 'Test_pos_patient': 'positive'})
print(df)

patient_stats = df.groupby('Patient_ID').agg({'predicted positive': 'sum', 'Total Count': 'sum', 'ML': 'first'})
patient_stats['Positives per 5M RBC'] = patient_stats['predicted positive'] / (patient_stats['Total Count'] / (5*10**6))
patient_stats.to_csv('patient_stats.csv')


# calculate the STD of the positives per 5M RBC for all ML == 'negative'
negative_stats = patient_stats[patient_stats['ML'] == 'negative']
print(negative_stats)
print("FPs / uL: ",negative_stats['Positives per 5M RBC'].std())