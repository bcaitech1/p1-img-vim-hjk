# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              
---
### Install Requirements
- `pip install -r requirements.txt`
---
### Prepare data label
- `python labeling.py --config config_name`
- You can use various data preprocessing to label and effectively manage data.
- Please refer to the label_config.yml for details.
- You can check the data analysis result example in file `label_anaylsis.ipynb`.
---
### Check data augmentation
- check_augmentation.ipynb
- You can see the applied augmentation.
---
### Training
- `python main.py --config config_name`
- You can choose 3 types model.
    - MultiConvModel
    - MultiSampleDropOut
    - MultiFCModel
- Please refer to the config.yml for details.
---
### Inference
- `python inference.py`
- It includes two tasks.
    - Ensemble(soft voting)
    - Test time augmentation


