import os
import sys
import yaml
import argparse
import pandas as pd

from glob import glob
from tqdm import tqdm
from pprint import pprint


# argparser
parser = argparse.ArgumentParser(description='Create labeled .csv file')
parser.add_argument('--config', required=False, default='Base', help='Enter the config name to apply.')
'''
parser.add_argument('--kfold', required=False, default=0, help='split dataset for k fold cv')
parser.add_argument('--modify_gender_ambiguity', required=False, default=False, help='modify ambiguous data')
parser.add_argument('--age_filter', required=False, default=60, help='you can adjust the maximum age value to under 60')
'''

args = parser.parse_args()
config_name = args.config
'''
kfold = int(args.kfold)
age_filter = int(args.age_filter)
'''

# Set config
with open("labeling_config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)[config_name]
    
kfold = cfg['kfold']
stratified = cfg['stratified']
modify_gender_ambiguity = cfg['modify_gender_ambiguity']
age_filter = cfg['age_filter']
    
# Thank you for T1034's EDA.
gender_abmigous_data_list = ['004432_male_Asian_43', '001498-1_male_Asian_23', '006359_female_Asian_18',
                             '006360_female_Asian_18', '006361_female_Asian_18', '006362_female_Asian_18',
                             '006363_female_Asian_18', '006364_female_Asian_18']

data_dir = '../input/data/train'
df_data = pd.read_csv(f'../input/data/train/train.csv')
image_path = '../input/data/train/images/'
save_path = f'../label/{config_name}'
kfold_save_path = f'{save_path}/kfold_label/'

os.makedirs(save_path, exist_ok=True)

num_person = len(glob(os.path.join(image_path, '*')))

# Show Config
print('\n\n======================Config=====================')
pprint(cfg)
print('=================================================\n')

def split_data(kfold, label_df, kfold_save_path):
    os.makedirs(f'{kfold_save_path}/train', exist_ok=True)
    os.makedirs(f'{kfold_save_path}/val', exist_ok=True)
    fraction = 1 / kfold
    seg = int(num_person * fraction)
    print('================K-Fold Information===============')
    for i in range(kfold):
        trll = 0
        trlr = 7 * i * seg
        vall = trlr
        valr = 7 * (i * seg + seg)
        trrl = valr
        trrr = 7 * num_person
        
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        
        print("train indices: [%d,%d),[%d,%d), validation indices: [%d,%d)" 
               % (trll,trlr,trrl,trrr,vall,valr))
        
        train_df = pd.concat([label_df[trll:trlr], label_df[trrl:trrr]])
        val_df = label_df[vall:valr]
        
        print(f'train length : {len(train_df)}, validation length : {len(val_df)}\n')
        
        train_df.to_csv(f'{kfold_save_path}/train/{i}_fold_train_label.csv')
        print(f'{i}_fold_train_label.csv has been saved in {kfold_save_path}train.')
        val_df.to_csv(f'{kfold_save_path}/val/{i}_fold_val_label.csv')    
        print(f'{i}_fold_val_label.csv has been saved in {kfold_save_path}val.')
        print('-------------------------------------------------')
    print('=================================================')

    
def make_balanced_df(kfold, label_df):
    label_list = []
    
    for g in range(2):
        for a in range(3):
            condition = (label_df['gender'] == g) & (label_df['age'] == a)
            label_list.append(label_df[condition].reset_index(drop=True))
    
    balanced_df = pd.DataFrame()
    index_list = []
    remainder_list = []
    
    for df in label_list:
        index_list.append(len(df) // 7 // kfold * 7)
        remainder_list.append(len(df) // 7 % kfold)
        
    for k in range(kfold):
        for idx in range(len(label_list)):            
            if len(label_list[idx]) >= index_list[idx]:        
                if remainder_list[idx] > 0:
                    end_index = index_list[idx] + 7
                    remainder_list[idx] -= 1
                else:
                    end_index = index_list[idx]
                # print(f'End index : {end_index}')
                balanced_df = pd.concat([balanced_df, label_list[idx].iloc[0:end_index]])
                # print(f'Len(balanced_df) : {len(balanced_df)}')
                label_list[idx].drop(list(range(end_index)), inplace=True)
                label_list[idx].reset_index(drop=True, inplace=True)
                # print(f'Len(label_list[idx]) : {len(label_list[idx])}\n')
                
    return balanced_df
    
    
class MaskLabels:
    mask = 0
    incorrect = 1
    normal = 2

    
class GenderLabels:
    male = 0
    female = 1

    
class AgeGroup:
    map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < age_filter else 2
    
    
_file_names = {
        "incorrect_mask.*": MaskLabels.incorrect,
        "mask1.*": MaskLabels.mask,
        "mask2.*": MaskLabels.mask,
        "mask3.*": MaskLabels.mask,
        "mask4.*": MaskLabels.mask,
        "mask5.*": MaskLabels.mask,
        "normal.*": MaskLabels.normal
    }

label_df = pd.DataFrame({'class' : [i for i in range(0, num_person * 7)],
                         'mask' : ['None'] * (num_person * 7),
                         'gender' : ['None'] * (num_person * 7),
                         'age' : ['None'] * (num_person * 7),
                         'image_path' : ['None'] * (num_person * 7)
                       })

for i in tqdm(range(num_person)):
    for j, (file_name, mask_label) in enumerate(_file_names.items()):
        # Gender Label
        gender_label = getattr(GenderLabels, df_data.loc[i, 'gender'])       
        label_df.loc[7 * i + j, 'gender'] = gender_label
        if modify_gender_ambiguity:
            if df_data.loc[i, 'path'] in gender_abmigous_data_list:
                label_df.loc[7 * i + j, 'gender'] = 0 if gender_label == 1 else 1
                
        # Age Label
        age_label = AgeGroup.map_label(int(df_data.loc[i, 'age']))
        label_df.loc[7 * i + j, 'age'] = age_label
        
        # Mask Label, File Path
        label_df.loc[7 * i + j, 'image_path'] = glob(os.path.join(
            image_path, str(df_data.loc[i, 'path']), file_name))[0]
        label_df.loc[7 * i + j, 'mask'] = mask_label
        
        # Class Label
        label_df.loc[7 * i + j, 'class'] = int(mask_label * 6 + gender_label * 3 + age_label)

print('\n=====================Head-14=====================')
print(label_df.head(14))
print('=================================================')

# Check None
for column in label_df.columns:
    if label_df[column].isin(['None']).any():
        problem_df = label_df[label_df[column].isin(['None'])]
        print('\n=====Problem id=====\n')
        print(problem_df)
        sys.exit()
else:
    print(f'\n', '*' * 10, 'All data was successfully entered.', '*' * 10, '\n')

label_df.to_csv(f'{save_path}/whole_label.csv')
print('\n', '*' * 10, f'whole_label.csv has been saved in {save_path}.', '*' * 10, '\n')
    
if kfold > 0:    
    os.makedirs(kfold_save_path, exist_ok=True)
    if stratified:
        label_df = make_balanced_df(kfold, label_df)
    split_data(kfold, label_df, kfold_save_path)