import pandas as pd
from google.cloud import storage
import os
from io import StringIO

import analysis

# BLOB is an acronym for "Binary Large Object". It's a data type that stores binary data, such as images, videos, and audio.
def get_csv_from_gcs(bucket_name, source_blob_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_text()
    return pd.read_csv(StringIO(data), low_memory=False)

asec_data = None

BUCKET_NAME = os.environ.get("BUCKET_NAME")

# if we're in the cloud environment get data from GCS, else get it locally

print(f"Loading merged ASEC data from {BUCKET_NAME if BUCKET_NAME else 'LOCAL'}...")
if BUCKET_NAME:
    asec_data = get_csv_from_gcs(BUCKET_NAME, 'merged_asec.csv')
    fam_data = get_csv_from_gcs(BUCKET_NAME, 'merged_fam.csv')
else:
    asec_data = pd.read_csv('data/output/merged/merged_asec.csv')
    fam_data = pd.read_csv('data/output/merged/merged_fam.csv')
print(f"Loaded ASEC data; Shape: {asec_data.shape}")

permutation_importance = analysis.PermutationImportance(asec_data)
cross_sectional_regression = analysis.CrossSectionalRegression(asec_data)
quantile = analysis.Quantile(fam_data)