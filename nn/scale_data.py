import numpy as np 
import pandas as pd
import math
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# TODO: add handling for genes being out of order and tissue names being weird

# columns: everything, excluse sex and tissue, only immune gene 
# df = pd.read_csv('../ProcessedData/SRR_AllGex_Normal.csv')
def scale_data(filepath, column_filter, scaler, split=.8, random_state=42):

    df = pd.read_csv(filepath)
    print(df.head(5))

    # drop unrelated columns TODO: these won't be the same for each user
    drop_list = ['Sample_run', 'LoadDate', 'spots', 'bases', 'avgLength', 'size_MB', 'Experiment', 'LibraryStrategy', 'LibraryLayout', 'Platform', 'SRAStudy', 'BioProject', 'Sample', 'BioSample', 'Notes', 'GSM', 'NA']
    df.drop(drop_list, axis=1, inplace=True)

    # filter columns based on user specification 
    if column_filter == 'all':
        
        # one-hot encoding for sex and tissue
        index = df.columns.get_loc('Sex')
        encoded = pd.get_dummies(df['Sex'], prefix='Sex')
        df = df.drop('Sex', axis=1)
        for i, col in enumerate(encoded.columns):
            df.insert(index + i, col, encoded[col])

        index = df.columns.get_loc('Tissue')
        encoded = pd.get_dummies(df['Tissue'], prefix='Tissue')
        df = df.drop('Tissue', axis=1)
        for i, col in enumerate(encoded.columns):
            df.insert(index + i, col, encoded[col])

        print(df)

    elif column_filter == 'exclude_sex_tissue':

        df = df.drop(['Sex', 'Tissue'], axis=1)
        print(df)

    elif column_filter == 'immune':

        immune_gene_list = '../ProcessedData/SRR_AllGex_Normal.csv'
        with open(immune_gene_list, encoding='utf-8') as f:
            kept_features = [line.rstrip('\n') for line in f]
            print(len(kept_features))
            kept_features.append('Age')
            df = df[kept_features]
            print(df) 
    else:
        print(f'invalid column filter: {column_filter}')
        quit()

    X = df.drop(columns=['Age'])
    y = df['Age']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=random_state)

    if scaler == 'none':
        X_train_scaled = X_train.astype('float32')
        X_test_scaled = X_test.astype('float32')
    elif scaler == 'standard': # StandardScaler from sklearn
        scaler = StandardScaler().set_output(transform='pandas')
        X_train_scaled = scaler.fit_transform(X_train) # scales all col.umns except 'Notes'
        X_test_scaled = scaler.transform(X_test)
    elif scaler == 'minMax':
        scaler = MinMaxScaler(feature_range=(0,1)).set_output(transform='pandas')
        X_train_scaled = scaler.fit_transform(X_train) # scales all col.umns except 'Notes'
        X_test_scaled = scaler.transform(X_test)       
    elif scaler == 'fractionOfMax':
        maxes = X_train.max().tolist()
        for i, m in enumerate(maxes): # avoids division by 0
            if m == 0: maxes[i] = 1
        X_train_scaled = (X_train.values / maxes).astype('float32')
        X_test_scaled = (X_test.values / maxes).astype('float32')
    else:
        print(f'invalid encoding: {scaler}')
        quit()

    X_train_scaled['Age'] = y_train
    X_test_scaled['Age'] = y_test
    X_train_scaled.to_csv(f'../ProcessedData/ScaledData/{os.path.basename(filepath)}_train_{column_filter}_{scaler}.csv')
    X_test_scaled.to_csv(f'../ProcessedData/ScaledData/{os.path.basename(filepath)}_test_{column_filter}_{scaler}.csv')