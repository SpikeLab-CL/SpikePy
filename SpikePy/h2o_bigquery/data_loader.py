from google.cloud import bigquery
from google.cloud import storage
import uuid
import h2o
import logging
import tempfile
import shutil
import subprocess
import sys

class H2OBigQueryLoader():
    """Conector used to load data from H2O directly into BigQuery
    Arguments:
        project_id : str used to specify the GCP project.
    Raises:
        ValueError: if project_id is None
    """
    def __init__(self, project_id=None):
        if project_id == None:
            raise ValueError("No project_id specified!")
        self.project_id = project_id
        self.bigquery_client = bigquery.Client(project=self.project_id)
        self.storage_client = storage.Client(project=self.project_id)
    
    def from_query(self, query=None, allow_large_results=False,
                                    destination_dataset=None, destination_table=None):
        """Loads data from BigQuery directly into H2O.
        Arguments:
            query: str query to run in BigQuery.
            allow_large_results: bool If set in True, allow to download big datasets
            destination_dataset: str destination dataset used if allow_large_results is set True
                                 if is set None, then a temporal dataset is created.
            destination_table: str destination table used if allow_large_results is set True
                                 if is set None, then a temporal table is created.
        """
        if query == None:
            raise ValueError("No query provided")
        if type(query) != str:
            raise TypeError("Query must be <type 'str'> got {0}".format(type(query)))
        if allow_large_results == False:
            query_job = self.bigquery_client.query(query=query)
            job_results = self._check_query_job(query_job)
            return h2o.H2OFrame(job_results.to_dataframe())
        else:
            delete_dataset = False
            if (destination_dataset == None or destination_table == None):
                destination_dataset = "temp_dataset_{0}".format(uuid.uuid1(5))
                destination_table = "temp_table_{0}".format(uuid.uuid1(5))
                delete_dataset = True

            query_job_config = bigquery.QueryJobConfig()
            query_job_config.create_disposition = "CREATE_IF_NEEDED"
            query_job_config.write_disposition = "WRITE_TRUNCATE"
            query_job_config.destination = "{project}.{dataset}.{table}".format(project=self.project_id,
                                                                                dataset=destination_dataset,
                                                                                table=destination_table)
            query_job = self.bigquery_client.query(query=query, job_config=query_job_config)
            job_results = self._check_query_job(query_job)
            table_ref = job_results.destination
            bucket = self._create_temporal_bucket()
            files_uri = self._export_result_to_storage(table_ref=table_ref, destination_bucket=bucket)
            temp_folder, files_regex = self._download_files_from_storage(files_uri=files_uri)
            try:
                loaded_data = h2o.import_file(path=temp_folder+"/",pattern = ".*\.csv.gz")
                self._remove_temporal_bucket(bucket)
                self._remove_temp_folder(path=temp_folder)
                dataset_ref = self.bigquery_client.dataset(destination_dataset)
                dataset = bigquery.Dataset(dataset_ref)
                if delete_dataset:
                    self.bigquery_client.delete_dataset(dataset)
                return loaded_data
            except Exception as e:
                logging.error(e)

    def _remove_temp_folder(self, path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            logging.error("Error deleting the files from {0}".format(path))
    
    def _export_result_to_storage(self, table_ref, destination_bucket):
        files_uris = "gs://{bucket}/{file}_*.csv.gz".format(bucket=destination_bucket.name,
                                                            file=table_ref.table_id)
        extract_job_config = bigquery.ExtractJobConfig()
        extract_job_config.compression = "gzip"
        extract_job_config.destination_format = "CSV"
        extract_job_config.field_delimiter = ";"
        extract_job = self.bigquery_client.extract_table(source=table_ref,
                                                         destination_uris=files_uris,
                                                         location="US",
                                                         job_config=extract_job_config)
        extract_job_result = extract_job.result()
        if(extract_job_result.errors != None):
                raise RuntimeError(extract_job_result.errors)
        return files_uris

    def _download_files_from_storage(self, files_uri):
        files_regex = files_uri.split("/")[-1]
        try:
            temp_folder = tempfile.mkdtemp()
            subprocess.check_call(['gsutil','-q','cp', files_uri, temp_folder], stderr=sys.stdout)
        except Exception as e:
            raise RuntimeError("Error while downloaded files from {0}".format(files_uri))
        return temp_folder, files_regex

    def _create_temporal_bucket(self):
        temporal_bucket_name = "temp_spike_{0}".format(uuid.uuid1(10))
        bucket = storage.Bucket(client=self.storage_client, name=temporal_bucket_name)
        bucket.location = "us"
        bucket = self.storage_client.create_bucket(bucket)
        return bucket

    def _remove_temporal_bucket(self, bucket):
        try:
            bucket.delete(force=True, client=self.storage_client)
        except Exception as e:
            logging.warning("Failed to delete the bucket: {0}".format(bucket.name))

    def _check_query_job(self, query_job):
        if(query_job.done()):
            if(query_job.errors != None):
                raise RuntimeError(query_job.errors)
        return query_job