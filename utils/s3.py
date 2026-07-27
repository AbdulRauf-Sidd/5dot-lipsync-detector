import boto3
from botocore.exceptions import ClientError
from config.project_config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, AWS_BUCKET_NAME


def _client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )


def upload_file(local_path: str, key: str, content_type: str = "application/octet-stream") -> str:
    _client().upload_file(
        local_path, AWS_BUCKET_NAME, key,
        ExtraArgs={"ContentType": content_type}
    )
    return key


def upload_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    _client().put_object(Bucket=AWS_BUCKET_NAME, Key=key, Body=data, ContentType=content_type)
    return key


def download_file(key: str, local_path: str) -> None:
    _client().download_file(AWS_BUCKET_NAME, key, local_path)


def delete_file(key: str) -> None:
    try:
        _client().delete_object(Bucket=AWS_BUCKET_NAME, Key=key)
    except ClientError:
        pass


def presigned_url(key: str, expires_in: int = 86400) -> str:
    """Returns a pre-signed GET URL valid for `expires_in` seconds (default 24 h)."""
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": AWS_BUCKET_NAME, "Key": key},
        ExpiresIn=expires_in,
    )
