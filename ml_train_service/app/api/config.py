import boto3
import os
# from dotenv import load_dotenv
# # OR, explicitly providing path to '.env'
# from pathlib import Path  # Python 3.6+ only
# env_path = Path('.') / '.env'
# load_dotenv(dotenv_path=env_path)

PATH_PKG = os.path.dirname(os.path.abspath(__file__))
# PATH_PKG = "."
FEATURE_PATH = os.path.join(PATH_PKG, "resources/features.json")
DATA_PATH = os.path.join(PATH_PKG, "data")
MODEL_PATH = os.path.join(PATH_PKG, "model")
RESULT_PATH = os.path.join(PATH_PKG, "result")



if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

# aws_access_key_id=
# aws_secret_access_key=
# region_name=
# folder=

session = boto3.Session(
    aws_access_key_id=os.environ["aws_access_key_id"],
    aws_secret_access_key=os.environ["aws_secret_access_key"],
    region_name=os.environ["region"])

s3_resource = session.resource('s3')
s3_client = session.client('s3')

bucket_name = os.environ["bucket_name"]
sagemaker_role = os.environ['sagemaker_role']
