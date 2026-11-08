import os

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from typing import Optional

S3_IP = os.getenv("IP")
S3_PORT = os.getenv("MINIO_PORT")
S3_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
S3_REGION = "Moscow"

S3_URL = f"http://{S3_IP}:{S3_PORT}"


def get_s3_client():
    """S3 stroage client"""
    session = boto3.session.Session()
    return session.client(
        service_name="s3",
        endpoint_url=S3_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
        config=Config(signature_version="s3v4"),
    )


def load_text_from_s3(bucket_name: str, file_name: str) -> str:
    """
    Dowanload text file from storage:bucket:name. Return as string
    """
    s3_client = get_s3_client()
    response = s3_client.get_object(Bucket=bucket_name, Key=file_name)
    return response["Body"].read().decode("utf-8")


def save_text_to_s3(bucket_name: str, file_name: str, text_content: str) -> None:
    """
    Save text to storage:bucket:filename.txt.
    """
    s3_client = get_s3_client()
    encoded_text = text_content.encode("utf-8")
    s3_client.put_object(
        Bucket=bucket_name, Key=file_name, Body=encoded_text, ContentType="text/plain"
    )

def generate_temporary_url(bucket_name: str, object_key: str,
                           expiration_seconds: int = 600) -> Optional[str]:
    """
    Generate tmp url for storage:bucket:name. Return as string
    """
    s3_client = get_s3_client()
    try:
        response: str = s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': bucket_name,
                'Key': object_key
            },
            ExpiresIn=expiration_seconds
        )
        return response
    except ClientError as e:
        print(f"Error in url generation occured: {e}")
        return None