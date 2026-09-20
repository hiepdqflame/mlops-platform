import os

import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3', endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'])
try:
    s3.head_bucket(Bucket='mlflow')
except ClientError as exc:
    if exc.response['Error']['Code'] not in ('404', 'NoSuchBucket'):
        raise
    s3.create_bucket(Bucket='mlflow')
s3.put_bucket_versioning(Bucket='mlflow', VersioningConfiguration={'Status': 'Enabled'})
print('MinIO bucket mlflow ready; object versioning enabled')
