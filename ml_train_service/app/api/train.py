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
    eli5.show_weights(perm, feature_names=features)

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
    features = args.feature_names

    best_model = ModelCheckpoint(os.path.join(base_path,"best.h5"), monitor='val_loss', verbose=1, save_best_only=True)

    csv_logger = CSVLogger(os.path.join(base_path,"history.csv"), append=True, separator=';')

    callbacks = [best_model,csv_logger]


    seed = 7
    np.random.seed(seed)
    estimator = KerasRegressor(build_fn=model, validation_split=0.2, verbose=1, epochs=10, batch_size=32,
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

