import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
from tensorflow.keras import layers
import argparse
import os
import numpy as np
import boto3
import json
import subprocess
import sys


features = {"feature_names": ["laXMax", "laYMax", "laZMax", "normlaMax", "aaXMax", "aaYMax", "aaZMax", "normaaMax", "avXMax", "avYMax", "avZMax", "normavMax", "laXMin", "laYMin", "laZMin", "normlaMin", "aaXMin", "aaYMin", "aaZMin", "normaaMin", "avXMin", "avYMin", "avZMin", "normavMin", "laX_int", "laY_int", "laZ_int", "normla_int", "aaX_int", "aaY_int", "aaZ_int", "normaa_int", "avX_int", "avY_int", "avZ_int", "normav_int", "abs_laX_int", "abs_laY_int", "abs_laZ_int", "abs_normla_int", "abs_aaX_int", "abs_aaY_int", "abs_aaZ_int", "abs_normaa_int", "abs_avX_int", "abs_avY_int", "abs_avZ_int", "abs_normav_int", "exp_laX_SR_max", "exp_laY_SR_max", "exp_laZ_SR_max", "exp_normla_SR_max", "exp_aaX_SR_max", "exp_aaY_SR_max", "exp_aaZ_SR_max", "exp_normaa_SR_max", "exp_avX_SR_max", "exp_avY_SR_max", "exp_avZ_SR_max", "exp_normav_SR_max", "exp_laX_SR_min", "exp_laY_SR_min", "exp_laZ_SR_min", "exp_normla_SR_min", "exp_aaX_SR_min", "exp_aaY_SR_min", "exp_aaZ_SR_min", "exp_normaa_SR_min", "exp_avX_SR_min", "exp_avY_SR_min", "exp_avZ_SR_min", "exp_normav_SR_min", "exp_laX_10SR_max", "exp_laY_10SR_max", "exp_laZ_10SR_max", "exp_normla_10SR_max", "exp_aaX_10SR_max", "exp_aaY_10SR_max", "exp_aaZ_10SR_max", "exp_normaa_10SR_max", "exp_avX_10SR_max", "exp_avY_10SR_max", "exp_avZ_10SR_max", "exp_normav_10SR_max", "exp_laX_10SR_min", "exp_laY_10SR_min", "exp_laZ_10SR_min", "exp_normla_10SR_min", "exp_aaX_10SR_min", "exp_aaY_10SR_min", "exp_aaZ_10SR_min", "exp_normaa_10SR_min", "exp_avX_10SR_min", "exp_avY_10SR_min", "exp_avZ_10SR_min", "exp_normav_10SR_min", "exp_laX_100SR_max", "exp_laY_100SR_max", "exp_laZ_100SR_max", "exp_normla_100SR_max", "exp_aaX_100SR_max", "exp_aaY_100SR_max", "exp_aaZ_100SR_max", "exp_normaa_100SR_max", "exp_avX_100SR_max", "exp_avY_100SR_max", "exp_avZ_100SR_max", "exp_normav_100SR_max", "exp_laX_100SR_min", "exp_laY_100SR_min", "exp_laZ_100SR_min", "exp_normla_100SR_min", "exp_aaX_100SR_min", "exp_aaY_100SR_min", "exp_aaZ_100SR_min", "exp_normaa_100SR_min", "exp_avX_100SR_min", "exp_avY_100SR_min", "exp_avZ_100SR_min", "exp_normav_100SR_min", "num_peaks_laX", "num_peaks_laY", "num_peaks_laZ", "num_peaks_normla", "num_peaks_aaX", "num_peaks_aaY", "num_peaks_aaZ", "num_peaks_normaa", "num_peaks_avX", "num_peaks_avY", "num_peaks_avZ", "num_peaks_normav", "num_valleys_laX", "num_valleys_laY", "num_valleys_laZ", "num_valleys_normla", "num_valleys_aaX", "num_valleys_aaY", "num_valleys_aaZ", "num_valleys_normaa", "num_valleys_avX", "num_valleys_avY", "num_valleys_avZ", "num_valleys_normav", "peaks4_laX2", "peaks4_laX3", "peaks4_laX4", "peaks4_laX5", "peaks4_laY2", "peaks4_laY3", "peaks4_laY4", "peaks4_laY5", "peaks4_laZ2", "peaks4_laZ3", "peaks4_laZ4", "peaks4_laZ5", "peaks4_normla2", "peaks4_normla3", "peaks4_normla4", "peaks4_normla5", "peaks4_aaX2", "peaks4_aaX3", "peaks4_aaX4", "peaks4_aaX5", "peaks4_aaY2", "peaks4_aaY3", "peaks4_aaY4", "peaks4_aaY5", "peaks4_aaZ2", "peaks4_aaZ3", "peaks4_aaZ4", "peaks4_aaZ5", "peaks4_normaa2", "peaks4_normaa3", "peaks4_normaa4", "peaks4_normaa5", "peaks4_avX2", "peaks4_avX3", "peaks4_avX4", "peaks4_avX5", "peaks4_avY2", "peaks4_avY3", "peaks4_avY4", "peaks4_avY5", "peaks4_avZ2", "peaks4_avZ3", "peaks4_avZ4", "peaks4_avZ5", "peaks4_normav2", "peaks4_normav3", "peaks4_normav4", "peaks4_normav5", "valley4_laX2", "valley4_laX3", "valley4_laX4", "valley4_laX5", "valley4_laY2", "valley4_laY3", "valley4_laY4", "valley4_laY5", "valley4_laZ2", "valley4_laZ3", "valley4_laZ4", "valley4_laZ5", "valley4_normla2", "valley4_normla3", "valley4_normla4", "valley4_normla5", "valley4_aaX2", "valley4_aaX3", "valley4_aaX4", "valley4_aaX5", "valley4_aaY2", "valley4_aaY3", "valley4_aaY4", "valley4_aaY5", "valley4_aaZ2", "valley4_aaZ3", "valley4_aaZ4", "valley4_aaZ5", "valley4_normaa2", "valley4_normaa3", "valley4_normaa4", "valley4_normaa5", "valley4_avX2", "valley4_avX3", "valley4_avX4", "valley4_avX5", "valley4_avY2", "valley4_avY3", "valley4_avY4", "valley4_avY5", "valley4_avZ2", "valley4_avZ3", "valley4_avZ4", "valley4_avZ5", "valley4_normav2", "valley4_normav3", "valley4_normav4", "valley4_normav5"]}

def model():
    model = tf.keras.Sequential([
        layers.Dense(300, input_dim=240, activation='relu', kernel_regularizer='l2'),
        layers.Dropout(.5),
        layers.Dense(100, activation='relu', kernel_regularizer='l2'),
        layers.Dropout(.5),
        layers.Dense(20, activation='relu', kernel_regularizer='l2'),
        layers.Dense(17030, activation='relu', kernel_regularizer='l2')])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mae'])

    return model

def permutation_importance(X,y,estimator,features,base_path):
    perm = PermutationImportance(estimator, random_state=1).fit(X, y)

    zip_iterator = zip(features, perm.feature_importances_.tolist())

    weight_dict = dict(zip_iterator)

    with open(os.path.join(base_path,'feature_importance.json'), 'w') as fp:
        json.dump(weight_dict, fp)


def _load_training_data(base_dir):
    X_train = np.load(os.path.join(base_dir, 'training.npy'))
    y_train = np.load(os.path.join(base_dir, 'training_label.npy'))
    return X_train, y_train


def _load_validation_data(base_dir):
    X_val = np.load(os.path.join(base_dir, 'validation.npy'))
    y_val = np.load(os.path.join(base_dir, 'validation_label.npy'))
    return X_val, y_val


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--bucket_name', type=str)
    parser.add_argument('--sm_model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":

    args, unknown = _parse_args()

    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_validation_data(args.train)

    client = boto3.client('s3')
    bucket = args.bucket_name
    client.put_object(Bucket=bucket, Key="models/base_model")


    # base_path = os.path.join(args.sm_model_dir, 'base_model')
    base_path = args.sm_model_dir

    best_model = ModelCheckpoint(os.path.join(base_path,"best.h5"), monitor='val_loss', verbose=1, save_best_only=True)

    csv_logger = CSVLogger(os.path.join(base_path,"history.csv"), append=True, separator=';')

    callbacks = [best_model,csv_logger]


    seed = 7
    np.random.seed(seed)
    estimator = KerasRegressor(build_fn=model, validation_split=0.2, verbose=0, epochs=10, batch_size=32,
                               callbacks=callbacks)
    estimator.fit(train_data, train_labels)
    # loaded_model = mdl.load_weights(os.path.join(args.sm_model_dir,"best.h5"))
    estimator.model.save(os.path.join(base_path,"model.h5"))

    #save feature importance
    permutation_importance(train_data,train_labels,estimator,features,base_path)


    client.upload_file(Filename=os.path.join(base_path,"model.h5"),
                       Bucket=bucket,
                       Key='models/base_model/model.h5')

    client.upload_file(Filename=os.path.join(base_path, "history.csv"),
                       Bucket=bucket,
                       Key='models/base_model/history.csv')

    client.upload_file(Filename=os.path.join(base_path, "feature_importance.json"),
                       Bucket=bucket,
                       Key='models/base_model/feature_importance.json')

