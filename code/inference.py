import os
import pandas as pd
import numpy as np
import timm
import albumentations
import albumentations.pytorch
import torch

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from model import *

test_dir = '/opt/ml/input/data/eval'

def test_time_augmentation(model, images):
    images = torch.split(images, 3, dim=1)
    for i in range(len(images)):
        if i == 0:
            preds_mask, preds_gender, preds_age = model(images[i])
        else:
            pred_mask, pred_gender, pred_age = model(images[i])
            preds_mask = torch.cat((preds_mask, pred_mask), dim=0)
            preds_gender = torch.cat((preds_gender, pred_gender), dim=0)
            preds_age = torch.cat((preds_age, pred_age), dim=0)
    
    return torch.mean(preds_mask, dim=0), torch.mean(preds_gender, dim=0), torch.mean(preds_age, dim=0)

class TestDataset(Dataset):
    def __init__(self, img_paths, augs, transform):
        self.img_paths = img_paths
        self.augs = augs
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        for i in range(len(augs) + 1):
            if i == 0:
                images = self.transform(image=np.array(image))['image']
            else:
                image = self.augs[i - 1](image=np.array(image))['image']
                images = torch.cat((images, self.transform(image=image)['image']), dim=0)
        return images

    def __len__(self):
        return len(self.img_paths)
    
# meta 데이터와 이미지 경로를 불러옵니다.
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

augs = [
    albumentations.HorizontalFlip(),
    # albumentations.CenterCrop(224, 224),
    # albumentations.ColorJitter(p=0.5)
]

transform = albumentations.Compose([            
            albumentations.Resize(384, 512),
            albumentations.Normalize(mean=(0.560, 0.524, 0.501), std=(0.233, 0.243, 0.245)),
            albumentations.pytorch.transforms.ToTensorV2()
])

dataset = TestDataset(image_paths, augs, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)

device = torch.device('cuda')


models = [
    MultiFCModel('dm_nfnet_f0', False),
    MultiFCModel('xception', False),
    MultiFCModel('resnext50_32x4d', False),
    MultiFCModel('xception65', False),
    MultiFCModel('efficientnet_b3', False)
]

weights = [
    './checkpoint/dm_nfnet_f0/dm_nfnet_f0_003epoch.pth',
    './checkpoint/xception/004.pth',
    './checkpoint/resnext50_32x4d/resnext50_32x4d_003epoch.pth',
    './checkpoint/xception65/xception65_003epoch.pth',
    './checkpoint/efficientnet_b3/efficientnet_b3_003epoch.pth'
]

all_mask_preds = []
all_gender_preds = []
all_age_preds = []

for model, weight in zip(models, weights):
    
    model.load_state_dict(torch.load(weight))
    model.to(device)
    model.eval()
    
    mask_preds = []
    gender_preds = []
    age_preds = []
    
    for images in tqdm(loader):        
        with torch.no_grad():
            images = images.to(device)
            
            pred_mask, pred_gender, pred_age = test_time_augmentation(model, images) 
            
            mask_preds.append(pred_mask.cpu().numpy())
            gender_preds.append(pred_gender.cpu().numpy())
            age_preds.append(pred_age.cpu().numpy())
            
    all_mask_preds.append(mask_preds)
    all_gender_preds.append(gender_preds)
    all_age_preds.append(age_preds)

if len(weights) == 1:
    model_name = weights[0].split('/')[-1][:-4]
else:
    model_name = f'{len(weights)}_models_ensemble'
    
os.makedirs(f'./numpy_output/{model_name}', exist_ok=True)
    
np.save(f'./numpy_output/{model_name}/all_mask_preds.npy', all_mask_preds)
np.save(f'./numpy_output/{model_name}/all_gender_preds.npy', all_gender_preds)
np.save(f'./numpy_output/{model_name}/all_age_preds.npy', all_age_preds)

pred_mask = np.mean(all_mask_preds, axis=0)
pred_gender = np.mean(all_gender_preds, axis=0)
pred_age = np.mean(all_age_preds, axis=0)

pred_mask = pred_mask.argmax(axis=-1)
pred_gender = pred_gender.argmax(axis=-1)
pred_age = pred_age.argmax(axis=-1)

all_predictions = pred_mask * 6 + pred_gender * 3 + pred_age
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, f'{model_name}_submission.csv'), index=False)
print('test inference is done!')