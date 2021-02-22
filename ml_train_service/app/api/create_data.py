import janitor
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from .config import *
from joblib import dump, load


def data_augment(df, std_range, InpFeat, mean=0):
    """Augment data by adding gaussian noise to each feature"""
    # Standard deviation of each features
    std_dict = InpFeat.std().to_dict()
    df_jitters = []

    # Converting all column to numeric
    df[df.columns.tolist()] = df[df.columns.tolist()].apply(pd.to_numeric)

    columns = df.columns.tolist()

    # Add gaussian noise to each column
    for std in std_range:
        df_jitter = df.copy()
        for column in columns:
            df_jitter[column] = df.jitter(
                column_name=column,
                dest_column_name=column,
                scale=std_dict.get(column) * std,
                clip=None,
                random_state=None)
        df_jitters.append(df_jitter)

    df_jitters.append(df)
    final_df = pd.concat(df_jitters)
    return final_df


def get_train_test_data(df,MPS_new):
    X_train, X_test, y_train, y_test = train_test_split(df, MPS_new, test_size=0.15, random_state=42)

    # Standardization
    scaler = StandardScaler()
    scaler.fit(X_train)

    # X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    X_train_final = data_augment(X_train, [0.01, 0.02], df, 0)
    y_train_final = np.concatenate((y_train, y_train, y_train), axis=0)

    # Standardizing training and validation set
    X_train_trans = scaler.transform(X_train_final)
    # X_valid_trans = scaler.transform(X_valid)
    X_test_trans = scaler.transform(X_test)

    # Natural Logarithm
    y_log = np.log1p(y_train)
    y_train_log = np.log1p(y_train_final)
    # y_valid_log = np.log1p(y_valid)

    scaler_mps = StandardScaler()
    scaler_mps.fit(y_log)

    y_train_trans = scaler_mps.transform(y_train_log)
    # y_valid_trans = scaler_mps.transform(y_valid_log)

    # Save standerdization model
    dump(scaler, os.path.join(MODEL_PATH, "scaler_X.joblib"))
    dump(scaler_mps, os.path.join(MODEL_PATH, "scaler_mps.joblib"))

    np.save(os.path.join(DATA_PATH, 'training_label.npy'), y_train_trans)
    np.save(os.path.join(DATA_PATH, 'training.npy'), X_train_trans)
    # np.save(os.path.join(DATA_PATH, 'validation.npy'), X_valid_trans)
    # np.save(os.path.join(DATA_PATH, 'validation_label.npy'), y_valid_trans)

    np.save(os.path.join(DATA_PATH, 'test.npy'), X_test_trans)
    np.save(os.path.join(DATA_PATH, 'test_label.npy'), y_test)

    # return X_train_trans,y_train_trans,X_valid_trans,y_valid_trans,X_test_trans,y_test

    return X_train_trans,y_train_trans,X_test_trans,y_test



