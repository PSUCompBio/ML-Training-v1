from sagemaker.tensorflow import TensorFlow
from .config import *
from .upload_data import upload
import botocore
from tensorflow.keras.models import load_model
import numpy as np
from joblib import load
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math
import json


def train_in_sagemaker():

    estimator = TensorFlow(
        entry_point="train.py",
        role=sagemaker_role,
        instance_count=1,
        instance_type='ml.m5.large',
        framework_version="2.2",
        py_version="py37",
        hyperparameters={"bucket_name":bucket_name},
    )

    try:
        data_path = "s3://" + bucket_name + "/sagemaker_training_data/base_model_data/"
        estimator.fit(data_path)
        return True
    except:
        return False


def download_model():
    KEY = 'models/base_model/model.h5'  # replace with your object key

    try:
        s3_resource.Bucket(bucket_name).download_file(KEY, os.path.join(MODEL_PATH, "model.h5"))
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


def find_95_percentile(array):
    """Return top 5 % element of a list"""

    array[::-1].sort()
    return array[:math.ceil(0.05 * len(array))]

def evaluate():
    """"""
    download_model()
    ml_model = load_model(os.path.join(MODEL_PATH, "model.h5"))
    scaler = load(os.path.join(MODEL_PATH, "scaler_mps.joblib"))
    test_data = np.load(os.path.join(DATA_PATH,"test.npy"))
    test_label = np.load(os.path.join(DATA_PATH,"test_label.npy"))

    y = ml_model.predict(test_data)
    # Exponentiang
    y_exp = np.expm1(y)
    y_result = scaler.inverse_transform(y_exp)

    r2 = r2_score(test_label, y_result)
    mse = mean_squared_error(test_label,y_result)

    mae = mean_absolute_error(test_label, y_result)

    ## 95% MPS
    mps_test = []
    mps_result = []

    for i in range(len(test_label)):
        mps_test.append(find_95_percentile(test_label[i]))

    for i in range(len(y_result)):
        mps_result.append(find_95_percentile(y_result[i]))

    mps_test = np.array(mps_test)
    mps_result = np.array(mps_result)

    r2_95 = r2_score(mps_test, mps_result)
    mse_95 = mean_squared_error(mps_test,mps_result)
    mae_95 = mean_absolute_error(mps_test,mps_result)

    result = {"mse": mse, "mae": mae, "r2": r2, "mse_95":mse_95, "mae_95":mae_95, "r2_95":r2_95}

    # Serializing json
    json_object = json.dumps(result, indent=4)

    # Writing to sample.json
    with open(os.path.join(RESULT_PATH,"test_result.json"), "w") as outfile:
        outfile.write(json_object)

    s3_client.upload_file(Filename=os.path.join(RESULT_PATH,"test_result.json"),
                       Bucket=bucket_name,
                       Key="models/base_ml/test_result.json")


def create_and_evaluate_model():
    try:
        print("Preprocessing and uploading training data")
        upload()
    except:
        raise
        return {"message":"Failed to create data","success":False}

    print("Training model")
    success = train_in_sagemaker()

    if success:
        try:
            print("Evaluating model")
            evaluate()
            return {"message":f"Results available at {os.path.join(bucket_name,'models/base_ml')}"}
        except:
            return {"message": "Failed to evaluate model", "success": False}

    else:
        return {"message":"Failed to train a model","success":False}

