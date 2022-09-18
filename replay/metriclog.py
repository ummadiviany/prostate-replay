import pandas as pd
import random
import os
from datetime import datetime
from json import dumps, loads, load, dump

col_names = ['Epoch',]
dnames = ['prostate158', 'isbi', 'promise12', 'decathlon']

for dname in dnames:
    for metric in ['Train Loss', 'Train Dice', 'Train HD']:
        col_names.append(f'{dname.upper()} {metric}')
        
for i in range(1, len(dnames)+1):
    for dname in dnames:
        col_names.append(f'{i}_{dname}_curr_dice')
        col_names.append(f'{i}_{dname}_curr_hd')
    

df = pd.DataFrame(columns=col_names, dtype=float, index=None)
df.set_index('Epoch', inplace=True)


for epoch in range(1, 26):
    df.loc[epoch] = [None]*(len(col_names)-1)


for i, dname in enumerate(dnames,1):
    for epoch in range(1, 25+1):
        
        log_metrics = {
            f'{dname.upper()} Train Loss': random.random(),
            f'{dname.upper()} Train Dice': random.random(),
            f'{dname.upper()} Train HD': random.random(),
        }
        
        for key,value in log_metrics.items():
            df.loc[epoch][key] = value
        
        
        for val_dname in dnames:
            log_metrics = {   
                    f'{i}_{val_dname}_curr_dice': random.random(),
                    f'{i}_{val_dname}_curr_hd': random.random(),
                }
                
            for key,value in log_metrics.items():
                df.loc[epoch][key] = value
config = {
    "Model" : "UNet2D",
    "Seqential Strategy" : "Raw/Naive Replay with 10% of dataset size in Episodic Memory",
    "Batch Training Strategy" : "A batch from current dataset and a batch from episodic memeory are stacked. One backward pass and paramenter update.",
#     "Train Input ROI size" : train_roi_size,
#     "Test Input size" : (1, 320, 320),
#     "Test mode" : f"Sliding window inference roi = {train_roi_size}",
    "Batch size" : "No of slices in original volume",
    "No of volumes per batch" : 1,
    "Optimizer" : "Adam",
    "Scheduler" : "CosineAnnealingLR",
    "Loss" : "DiceCELoss", 
    "Train Data Augumentations" : "Resize(256,256)",
    "Test Data Preprocess" : "Resize(256,256)",
    "Train samples" : {"Promise12" : 45, "ISBI" : 63, "Decathlon" : 25, "Prostate158" : 119},
    "Test Samples" : {"Promise12" : 5, "ISBI" : 16, "Decathlon" : 7, "Prostate158" : 20},
#     RandFlip, RandRotate90, RandGaussianNoise, RandGaussSmooth, RandBiasField, RandContrast
    "Pred Post Processing" : "KeepLargestConnectedComponent"
}
os.makedirs(f'metric_logs/{datetime.now().replace(microsecond=0)}', exist_ok=True)
dump(config, open(f'metric_logs/{datetime.now().replace(microsecond=0)}/config.json', 'w'))
df.to_csv(f'metric_logs/{datetime.now().replace(microsecond=0)}/metrics.csv')

# print(df.head())


