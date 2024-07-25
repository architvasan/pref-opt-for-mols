import os
import pandas as pd
from tqdm import tqdm

def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for it, line in enumerate(data):
            if it!=0:
                p = line.split(",")
                x.append((p[0]))
                y.append(float(p[1]))

    return x, y

path = 'scores'
dir_list = os.listdir(path)

smi_dat, dock_dat = Read_Two_Column_File(f'{path}/{dir_list[0]}')
#print(smi_dat)
#df = pd.read_csv(f'{path}/{dir_list[0]}')

for i in tqdm(range(1, len(dir_list))):
    smi_dat_n, dock_dat_n = Read_Two_Column_File(f'{path}/{dir_list[i]}')
    smi_dat.extend(smi_dat_n)
    dock_dat.extend(dock_dat_n)
    #df = pd.concat([df, pd.read_csv(f'{path}/{dir_list[i]}')])

df = pd.DataFrame({'smiles': smi_dat, 'scores': dock_dat})
df.drop_duplicates().to_csv('Merged_scores.csv', index=False)


if False:
    path = 'scores'
    dir_list = os.listdir(path)
    
    df = pd.read_csv(f'{path}/{dir_list[0]}')
    
    for i in tqdm(range(1, len(dir_list))):
        df = pd.concat([df, pd.read_csv(f'{path}/{dir_list[i]}')])
    
    
    df.drop_duplicates().to_csv('Merged_scores.csv', index=False)
