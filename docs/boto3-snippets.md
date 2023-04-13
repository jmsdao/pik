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
# Download a singular .csv file into IO buffer
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
# Download a singular .pkl file into IO buffer
buffer = io.BytesIO()
s3.download_fileobj(
    Bucket="BUCKET_NAME",
    Key="path/to/key/hidden_states.pkl",
    Fileobj=buffer,
)
buffer.seek(0)

data = pickle.load(buffer)
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
# Upload a singular .csv file from IO buffer
buffer = io.StringIO()
df.to_csv(buffer, index=False)

s3.put_object(
    Bucket="BUCKET_NAME",
    Key="path/to/key/blahblah.csv",
    Body=buffer.getvalue(),
)
```

```python
# Upload a singular .pkl file from IO buffer
pkl_obj = pickle.dumps(obj)  # len(pkl_obj) returns filesize in bytes
pkl_buffer = io.BytesIO(pkl_obj)

s3.upload_fileobj(
    Bucket="BUCKET_NAME",
    Key="path/to/key/blahblah.pkl",
    Fileobj=pkl_buffer,
)
```
