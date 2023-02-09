import logging
import uuid

from google.api_core.exceptions import Conflict, NotFound
from google.cloud import bigquery

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [%(name)s] [L%(lineno)d] [%(levelname)s][%(funcName)s] %(message)s "
    )
)
logger.addHandler(handler)
logger.propagate = False


class BQClient:
    def __init__(self, project, default_dataset=None):
        self.project = project
        self.default_dataset = default_dataset
        self.client = bigquery.Client(project=project)

    def _make_job_id(self, prefix=None):
        if prefix is not None:
            return str(prefix) + str(uuid.uuid4())
        else:
            return str(uuid.uuid4())

    def cancel_job(self, job_id):
        self.client.cancel_job(job_id)
        logger.info(f"Job id {job_id} was cancelled...")

    def create_dataset(self, dataset_id):
        try:
            dataset = self.client.dataset(dataset_id)
            dataset.location = "US"
            dataset = self.client.create_dataset(dataset)
            logger.info(f"Created dataset {self.client.project}.{dataset.dataset_id}")
        except Conflict:
            logger.info(f"{self.client.project}.{dataset.dataset_id} already exists.")

    def delete_dataset(self, dataset_id, delete_contents=False):
        try:
            dataset = self.client.get_dataset(dataset_id)
            dataset.location = "US"
            dataset = self.client.delete_dataset(
                dataset, delete_contents=delete_contents
            )
            logger.info(f"dataset: {self.client.project}.{dataset_id} was deleted.")
        except NotFound:
            logger.info(f"dataset: {self.client.project}.{dataset_id} not found.")

    def exist_table(self, dataset_id, table_id):
        ref = self.client.dataset(dataset_id).table(table_id)
        try:
            self.client.get_table(ref)
            logger.info(f"table: {dataset_id}.{table_id} exists.")
            return True
        except NotFound:
            logger.info(f"table: {dataset_id}.{table_id} not found.")
            return False

    def delete_table(self, dataset_id, table_id):
        if not self.exist_table(dataset_id, table_id):
            return

        ref = self.client.dataset(dataset_id).table(table_id)
        self.client.delete_table(ref)
        logger.info(f"table: {dataset_id}.{table_id} was deleted.")

    def create_table(self, dataset_id, table_id, setting, description=""):
        dataset_ref = self.client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)

        schema = [
            bigquery.SchemaField(field["name"], field["type"], mode=field["mode"])
            for field in setting["schema"]["fields"]
        ]
        table = bigquery.Table(table_ref, schema)
        table.description = description

        if setting.get("timePartitioning"):
            tp = bigquery.table.TimePartitioning()
            if setting.get("timePartitioning").get("field"):
                tp.field = setting["timePartitioning"]["field"]
            table.time_partitioning = tp

        if setting.get("clustering"):
            table.clustering_fields = setting["clustering"]["fields"]

        self.client.create_table(table)
        logger.info(
            f"Created table {table.project}.{table.dataset_id}.{table.table_id}"
        )

    def execute_query(self, query):
        if self.default_dataset is not None:
            default_dataset = self.project + "." + self.default_dataset
        else:
            default_dataset = None

        job_config = bigquery.job.QueryJobConfig(
            default_dataset=default_dataset,
            allow_large_results=True,
            dry_run=True,
        )
        job_config.use_legacy_sql = False
        job_id = self._make_job_id(prefix="execute_")
        try:
            query_job = self.client.query(query, job_id=job_id, job_config=job_config)
            logger.info(
                f"This query will process {query_job.total_bytes_processed / 1e9} GB."
            )
            job_config.dry_run = False
            query_job = self.client.query(query, job_id=job_id, job_config=job_config)
            query_job.result()
            logger.info("Executed query.")
        except KeyboardInterrupt as e:
            self.cancel_job(job_id)
            raise e

    def copy_table(self, src_project, src_dataset, tgt_dataset, table_id):
        if self.exist_table(tgt_dataset, table_id):
            self.delete_table(tgt_dataset, table_id)
        src_table_id = f"{src_project}.{src_dataset}.{table_id}"
        tgt_table_id = f"{self.project}.{tgt_dataset}.{table_id}"
        job_id = self._make_job_id(prefix="copy_")
        try:
            copy_job = self.client.copy_table(src_table_id, tgt_table_id, job_id=job_id)
            copy_job.result()
            logger.info("A copy of the table created.")
        except KeyboardInterrupt as e:
            self.cancel_job(job_id)
            raise e
