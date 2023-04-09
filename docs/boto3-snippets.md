```python
import boto3
import io
import pandas as pd

s3 = boto3.client("s3",
    aws_access_key_id="ACCESS_KEY_ID",
    aws_secret_access_key="SECRET_ACCESS_KEY",
)
```

```python
# Download a singular file to disk
s3.download_file(
    Bucket="BUCKET_NAME",
    Key="path/to/key/blahblah.csv",
    Filename="blahblah.csv"
)
```

```python
# Download a singular file into IO buffer
buffer = io.BytesIO()
s3.download_fileobj(
    Bucket="BUCKET_NAME",
    Key="path/to/key/blahblah.csv",
    Fileobj=buffer,
)
buffer.seek(0)

df = pd.read_csv(buffer)
```

```python
# Upload a singular file from disk
s3.upload_file(
    Bucket="BUCKET_NAME",
    Key="path/to/key/blahblah.csv",
    Filename="blahblah.csv",
)
```

```python
# Upload a singular file from IO buffer
buffer = io.StringIO()
df.to_csv(buffer, index=False)

s3.put_object(
    Bucket="BUCKET_NAME",
    Key="path/to/key/blahblah.csv",
    Body=buffer.getvalue(),
)
```
