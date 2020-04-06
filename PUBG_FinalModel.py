# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import gc
gc.enable()
# Any results you write to the current directory are saved as output.
pubg_train = pd.read_csv('../input/train_V2.csv')

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
    
pubg_train = reduce_mem_usage(pubg_train)

pubg_train_X = pubg_train.iloc[:,0:pubg_train.shape[1]-1]
pubg_train_Y = pubg_train.iloc[:,pubg_train.shape[1]-1]


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(pubg_train_X['matchType'])
pubg_train_X['matchType'] = le.transform(pubg_train_X['matchType'])

from sklearn.preprocessing import Imputer
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
pubg_train_Y = pubg_train_Y.values.reshape(len(pubg_train_Y), 1)
mean_imputer = mean_imputer.fit(pubg_train_Y)
pubg_train_Y = mean_imputer.transform(pubg_train_Y)

def rank_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop   and '_mean_rank' not in col  and '_median' not in col  and '_mean' not in col  and '_min' not in col and '_max' not in col]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    agg = agg.groupby('matchId')[features].rank(pct=True)
    return df.merge(agg, suffixes=['', '_mean_rank'], how='left', on=['matchId', 'groupId'])
 
def median_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop   and '_mean_rank' not in col  and '_median' not in col  and '_mean' not in col  and '_min' not in col and '_max' not in col]
    agg = df.groupby(['matchId', 'groupId'])[features].median()
    return df.merge(agg, suffixes=['', '_median'], how='left', on=['matchId', 'groupId'])

def mean_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop   and '_mean_rank' not in col  and '_median' not in col  and '_mean' not in col  and '_min' not in col and '_max' not in col]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    return df.merge(agg, suffixes=['', '_mean'], how='left', on=['matchId', 'groupId'])

def min_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop   and '_mean_rank' not in col  and '_median' not in col  and '_mean' not in col  and '_min' not in col and '_max' not in col]
    agg = df.groupby(['matchId','groupId'])[features].min()
    return df.merge(agg, suffixes=['', '_min'], how='left', on=['matchId', 'groupId'])

def max_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop   and '_mean_rank' not in col  and '_median' not in col  and '_mean' not in col  and '_min' not in col and '_max' not in col]
    agg = df.groupby(['matchId', 'groupId'])[features].max()
    return df.merge(agg, suffixes=['', '_max'], how='left', on=['matchId', 'groupId'])

def players_in_team(df):
    agg = df.groupby(['groupId']).size().to_frame('players_in_team')
    return df.merge(agg, how='left', on=['groupId'])

def players_joined(df):  
    df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
    return df

del pubg_train
gc.collect()
pubg_train=pd.DataFrame()
    
pubg_train_X = rank_by_team(pubg_train_X)
pubg_train_X = median_by_team(pubg_train_X)
pubg_train_X = mean_by_team(pubg_train_X)
pubg_train_X = max_by_team(pubg_train_X)
pubg_train_X = min_by_team(pubg_train_X)
pubg_train_X = players_in_team(pubg_train_X)
pubg_train_X = players_joined(pubg_train_X)
pubg_train_X = pubg_train_X.drop(columns=['Id','groupId','matchId'])

pubg_train_X = reduce_mem_usage(pubg_train_X)
gc.collect()

import lightgbm
categorical_features = ['matchType']
train_data = lightgbm.Dataset(pubg_train_X, label=pubg_train_Y.ravel(), categorical_feature=categorical_features)
parameters = {
    'metric': 'mae',
    'boosting': 'gbdt',
    'n_estimators': 10000,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'num_leaves': 80,
    'max_depth': 20,
    'verbose': 0
}
clf = lightgbm.train(parameters, train_data, 10000)

del [[pubg_train_X, pubg_train_Y]]
gc.collect()
pubg_train_X=pd.DataFrame()
pubg_train_Y=pd.DataFrame()



pubg_test = pd.read_csv('../input/test_V2.csv')
pubg_test = reduce_mem_usage(pubg_test)

pubg_test['matchType'] = le.transform(pubg_test['matchType'])
pubg_test = rank_by_team(pubg_test)
pubg_test = median_by_team(pubg_test)
pubg_test = mean_by_team(pubg_test)
pubg_test = max_by_team(pubg_test)
pubg_test = min_by_team(pubg_test)
pubg_test = players_in_team(pubg_test)
pubg_test = players_joined(pubg_test)

pubg_test = reduce_mem_usage(pubg_test)
gc.collect()

test_id = pubg_test['Id']
test_grp = pubg_test['groupId']
test_match = pubg_test['matchId']
pubg_test = pubg_test.drop(columns=['Id','groupId','matchId'])
y_pred=clf.predict(pubg_test)

#pubg_test['winPlacePerc'] = pubg_test.groupby('groupId')['winPlacePerc'].transform('mean')

df_sub = pd.DataFrame({'Id': test_id, 'winPlacePerc': y_pred})
df_test = pubg_test
df_test['Id'] = test_id
df_test['groupId'] = test_grp
df_test['matchId'] = test_match

# Restore some columns
df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

# Sort, rank, and assign adjusted ratio
df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()
df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
df_sub_group = df_sub_group.merge(
    df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    on="matchId", how="left")
df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)

df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
df_sub["winPlacePerc"] = df_sub["adjusted_perc"]

# Deal with edge cases
df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0
df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1

# Align with maxPlace
# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
subset = df_sub.loc[df_sub.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc

# Edge case
df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0
assert df_sub["winPlacePerc"].isnull().sum() == 0

df_sub['winPlacePerc'] = df_sub.groupby('groupId')['winPlacePerc'].transform('mean')
df_sub[["Id", "winPlacePerc"]].to_csv('LGBMOutput.csv',index=False)