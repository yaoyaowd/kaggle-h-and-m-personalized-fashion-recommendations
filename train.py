# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import gc
import pickle

import catboost
import numpy as np
import pandas as pd
import torch

from metric import mapk


dataset = '100'
class CFG:
    model_type = 'CatBoost'
    # candidates
    popular_num_items = 60
    popular_weeks = 1
    train_weeks = 6
    item2item_num_items = 24
    item2item_num_items_for_same_product_code = 12
    cooc_weeks = 32
    cooc_threshold = 150
    ohe_distance_num_items = 48
    ohe_distance_num_weeks = 20
    # features
    user_transaction_feature_weeks = 50
    item_transaction_feature_weeks = 16
    item_age_feature_weeks = 40
    user_volume_feature_weeks = 50
    item_volume_feature_weeks = 20
    user_item_volume_feature_weeks = 16
    age_volume_feature_weeks = 1
    
train = torch.load(f'input/{dataset}/train.pt')
valid = torch.load(f'input/{dataset}/valid.pt')
dataset_valid_all = torch.load(f'input/{dataset}/dataset_valid_all.pt')

# feature_columns = [c for c in valid.columns if c not in ['y', 'strategy', 'query_group', 'week']]
feature_columns = [
'user',
'item',
'age',
'product_type_no_idx',
'product_group_name_idx',
'graphical_appearance_no_idx',
'colour_group_code_idx',
'perceived_colour_value_id_idx',
'perceived_colour_master_id_idx',
'department_no_idx',
'index_code_idx',
'index_group_no_idx',
'section_no_idx',
'garment_group_no_idx',]

rank_features = [
'repurchase_week_rank',
'repurchase_volume_rank',
'pop_rank',
'age_popular_rank',
'same_product_code_item2item2_week_rank',
'same_product_code_item2item2_volume_rank',
'cat_volume',
'cat_volume_rank',
]

human_features = [
'user_price_mean',
'user_price_std',
'user_sales_channel_id_mean',
'user_sales_channel_id_std',
'item_price_mean',
'item_price_std',
'item_sales_channel_id_mean',
'item_sales_channel_id_std',
'age_mean',
'age_std',
'item_day_min',
'item_volume',
'user_day_min',
'user_volume',
'user_item_day_min',
'user_item_volume',
'age_volume',
]

print(feature_columns)

cat_feature_values = [c for c in feature_columns if c.endswith('idx')]
cat_features = [feature_columns.index(c) for c in cat_feature_values]
print(cat_feature_values, cat_features)

train_dataset = catboost.Pool(data=train[feature_columns], label=train['y'], group_id=train['query_group'], cat_features=cat_features)
valid_dataset = catboost.Pool(data=valid[feature_columns], label=valid['y'], group_id=valid['query_group'], cat_features=cat_features)

params = {
    'loss_function': 'YetiRank',
    'use_best_model': True,
    'one_hot_max_size': 300,
    'iterations': 5000,
}
model = catboost.CatBoost(params)
model.fit(train_dataset, eval_set=valid_dataset)

del train, valid, train_dataset, valid_dataset
gc.collect()
with open('output/model_for_validation.pkl', 'wb') as f:
    pickle.dump(model, f)

pred = dataset_valid_all[['user', 'item']].reset_index(drop=True)
pred['pred'] = model.predict(dataset_valid_all[feature_columns])

pred = pred.groupby(['user', 'item'])['pred'].max().reset_index()
pred = pred.sort_values(by=['user', 'pred'], ascending=False).reset_index(drop=True).groupby('user')['item'].apply(lambda x: list(x)[:12]).reset_index()

transactions = pd.read_pickle(f"input/{dataset}/transactions_train.pkl")
gt = transactions.query("week == 0").groupby('user')['item'].apply(list).reset_index().rename(columns={'item': 'gt'})
merged = gt.merge(pred, on='user', how='left')
merged['item'] = merged['item'].fillna('').apply(list)

merged.to_pickle(f'output/merged_{dataset}.pkl')

print('mAP@12:', mapk(merged['gt'], merged['item']))



'''
['user', 'item', 'repurchase_week_rank', 'repurchase_volume_rank', 'pop_rank', 'age_popular_rank', 'same_product_code_item2item2_week_rank', 'same_product_code_item2item2_volume_rank', 'cat_volume', 'cat_volume_rank', 'age', 'product_type_no_idx', 'product_group_name_idx', 'graphical_appearance_no_idx', 'colour_group_code_idx', 'perceived_colour_value_id_idx', 'perceived_colour_master_id_idx', 'department_no_idx', 'index_code_idx', 'index_group_no_idx', 'section_no_idx', 'garment_group_no_idx', 'user_price_mean', 'user_price_std', 'user_sales_channel_id_mean', 'user_sales_channel_id_std', 'item_price_mean', 'item_price_std', 'item_sales_channel_id_mean', 'item_sales_channel_id_std', 'age_mean', 'age_std', 'item_day_min', 'item_volume', 'user_day_min', 'user_volume', 'user_item_day_min', 'user_item_volume', 'age_volume']
['product_type_no_idx', 'product_group_name_idx', 'graphical_appearance_no_idx', 'colour_group_code_idx', 'perceived_colour_value_id_idx', 'perceived_colour_master_id_idx', 'department_no_idx', 'index_code_idx', 'index_group_no_idx', 'section_no_idx', 'garment_group_no_idx']
[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
'''