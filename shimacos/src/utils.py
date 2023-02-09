import polars as pl
from google.cloud import bigquery
from google.cloud.bigquery.job import QueryJobConfig
from queries.bq import BQClient


def read_gbq(query, project_id):
    client = bigquery.Client(project=project_id)
    df = (
        client.query(query)
        .to_arrow(progress_bar_type="tqdm", create_bqstorage_client=True)
        .to_pandas()
    )
    return df


def read_gbq_allow_large_results(query, dataset_id, table_id, project_id):
    bq = BQClient(project=project_id)
    bq.delete_table(dataset_id, table_id)
    job_config = QueryJobConfig(
        allow_large_results=True,
        destination=f"{project_id}.{dataset_id}.{table_id}",
    )
    client = bigquery.Client(project=project_id)
    df = (
        client.query(query, job_config=job_config)
        .to_arrow(progress_bar_type="tqdm", create_bqstorage_client=True)
        .to_pandas()
    )
    bq.delete_table(dataset_id, table_id)
    return df


def read_gbq_allow_large_results_polars(query, dataset_id, table_id, project_id):
    bq = BQClient(project=project_id)
    bq.delete_table(dataset_id, table_id)
    job_config = QueryJobConfig(
        allow_large_results=True,
        destination=f"{project_id}.{dataset_id}.{table_id}",
    )
    client = bigquery.Client(project=project_id)
    df = pl.from_pandas(
        client.query(query, job_config=job_config)
        .to_arrow(progress_bar_type="tqdm", create_bqstorage_client=True)
        .to_pandas()
    )
    bq.delete_table(dataset_id, table_id)
    return df
