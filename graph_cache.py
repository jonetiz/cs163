import plotly.io as pio
from google.cloud import storage
import os, json

BUCKET_NAME = os.environ.get("BUCKET_NAME")

def _gcloud_write(bucket_name, blob_name, data):
    """Write a JSON blob to GCS"""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_string(
        data=json.dumps(data),
        content_type='application/json'
    )

def _gcloud_read(bucket_name, blob_name):
    """Read a JSON blob from GCS"""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    json_string = blob.download_as_string()
    return json.loads(json_string)

def cache(figure, file_name):
    """Saves a Plotly `Figure` object in the cache with the specified `file_name`."""
    if BUCKET_NAME:
        data = pio.to_json(figure)
        _gcloud_write(BUCKET_NAME, file_name, json.loads(data))
    else:
        pio.write_json(figure, f'cache/{file_name}.json')

def get(file_name):
    """Returns a plotly figure object cached with the specified `file_name`."""
    if BUCKET_NAME:
        return _gcloud_read(BUCKET_NAME, file_name)
    
    return pio.read_json(f'cache/{file_name}.json')