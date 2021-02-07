from .preprocess import read_data
from .config import *
from .create_data import get_train_test_data

def upload_data():

    sub_folder = "sagemaker_training_data/base_model_data"
    s3_client.put_object(Bucket=bucket_name, Key=sub_folder)

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            s3_client.upload_file(os.path.join(root, file), bucket_name, os.path.join(sub_folder, file))


def upload_pickle():
    sub_folder = "models/base_model"
    s3_client.put_object(Bucket=bucket_name, Key=sub_folder)

    for root, dirs, files in os.walk(MODEL_PATH):
        for file in files:
            s3_client.upload_file(os.path.join(root, file), bucket_name, os.path.join(sub_folder, file))

def upload():
    try:
        features,targets = read_data()
        get_train_test_data(features,targets)
        upload_data()
        upload_pickle()
    except:
        raise


