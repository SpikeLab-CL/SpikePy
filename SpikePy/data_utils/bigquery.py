from google.cloud import bigquery
from google.auth import default
from uuid import uuid4
import pandas_gbq as pbq
import pandas as pd


class BigQueryOperator():
    """
    Operator to interact with BigQuery.
    Uses default credentials given by google.auth
    """
    def __init__(self, project_id=None, credentials=None):
        if credentials == None:
            credentials, default_project_id = default()
        if project_id == None:
            raise ValueError(
                "project_id is None or could not be determinate it")
        self.credentials = credentials
        self.project_id = project_id
        self.client = bigquery.Client(project=self.project_id,
                                      credentials=self.credentials)

    def run_query(self, query: str = None,
                  dataset: str = None,
                  table: str = None,
                  write_disposition: str = "WRITE_TRUNCATE") -> None:
        """
        Executes a query in BigQuery without bringing data to python.
        Good for creating and modifying table
        """
        valid_values = ['WRITE_TRUNCATE', 'WRITE_APPEND', 'WRITE_EMPTY']
        if write_disposition not in valid_values:
            raise ValueError("Invalid write_disposition, valid values: {0}"
                             .format(",".join(valid_values)))
        job_id = "metaflow_bq_query_job_{0}".format(str(uuid4()))
        job_config = bigquery.QueryJobConfig(
            create_disposition="CREATE_IF_NEEDED",
            destination="{project}.{dataset}.{table}".format(project=self.project_id,
                                                             dataset=dataset,
                                                             table=table),
            use_legacy_sql=False,
            write_disposition=write_disposition
        )
        query_job = self.client.query(
            query=query, job_config=job_config, job_id=job_id)
        query_job.result(max_results=0)

    def df_from_query(self, query: str = None, use_bqstorage_api: bool = True) -> pd.DataFrame:
        df = pbq.read_gbq(query=query,
                          use_bqstorage_api=use_bqstorage_api,
                          project_id=self.project_id)
        return df
