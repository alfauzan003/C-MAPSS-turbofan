"""Tiny wrappers over boto3 + pyarrow for parquet I/O against MinIO/S3.

Other modules:
    from pdm.storage import write_parquet, read_parquet, s3_uri
    uri = write_parquet(df, bucket="raw-data", key="snapshots/abc.parquet")
    df2 = read_parquet(uri)
"""

from __future__ import annotations

import io

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pdm.config import get_settings


def _s3_client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.minio_endpoint_url,
        aws_access_key_id=s.minio_access_key,
        aws_secret_access_key=s.minio_secret_key,
        region_name="us-east-1",
    )


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def write_parquet(df: pd.DataFrame, bucket: str, key: str) -> str:
    """Write `df` as parquet to s3://<bucket>/<key>. Returns the URI."""
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    _s3_client().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    return s3_uri(bucket, key)


def read_parquet(uri: str) -> pd.DataFrame:
    """Read a parquet file from an s3:// URI into a DataFrame."""
    assert uri.startswith("s3://"), f"Expected s3:// URI, got {uri}"
    bucket, key = uri[len("s3://") :].split("/", 1)
    obj = _s3_client().get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))
