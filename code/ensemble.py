import os
import numpy as np
import pandas as pd

test_dir = '/opt/ml/input/data/eval'
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))

all_mask_preds = np.load('./numpy_output/5_models_ensemble/all_mask_preds.npy')
all_gender_preds = np.load('./numpy_output/5_models_ensemble/all_gender_preds.npy')
all_age_preds = np.load('./numpy_output/5_models_ensemble/all_age_preds.npy')

all_mask_preds = np.delete(all_mask_preds, 0, axis=0)
all_mask_preds = np.delete(all_mask_preds, 0, axis=0)
all_gender_preds = np.delete(all_gender_preds, 0, axis=0)
all_gender_preds = np.delete(all_gender_preds, 0, axis=0)
all_age_preds = np.delete(all_age_preds, 0, axis=0)
all_age_preds = np.delete(all_age_preds, 0, axis=0)

pred_mask = np.mean(all_mask_preds, axis=0)
pred_gender = np.mean(all_gender_preds, axis=0)
pred_age = np.mean(all_age_preds, axis=0)

pred_mask = pred_mask.argmax(axis=-1)
pred_gender = pred_gender.argmax(axis=-1)
pred_age = pred_age.argmax(axis=-1)

all_predictions = pred_mask * 6 + pred_gender * 3 + pred_age
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(f'./{4}_submission.csv', index=False)
print('test inference is done!')