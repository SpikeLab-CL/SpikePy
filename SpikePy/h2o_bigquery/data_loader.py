from google.cloud import bigquery
from google.cloud import storage
import uuid
import h2o
import logging
import tempfile
import shutil
import subprocess
import sys
from datetime import datetime
import time

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
                destination_dataset = "temp_dataset_{0}".format(uuid.uuid1(5)).replace("-","_")
                destination_table = "temp_table_{0}".format(uuid.uuid1(5)).replace("-","_")
                self.bigquery_client.create_dataset(bigquery.Dataset(self.bigquery_client.dataset(destination_dataset)))
                delete_dataset = True

            query_job_config = bigquery.QueryJobConfig()
            query_job_config.create_disposition = "CREATE_IF_NEEDED"
            query_job_config.write_disposition = "WRITE_TRUNCATE"

            dataset_ref = bigquery.DatasetReference(project=self.project_id,dataset_id=destination_dataset)
            table_ref = bigquery.TableReference(dataset_ref=dataset_ref,table_id=destination_table)

            query_job_config.destination = table_ref
            query_job = self.bigquery_client.query(query=query, job_config=query_job_config)
            query_job.done()
            time.sleep(15)
            job_results = self._check_query_job(query_job)
            bucket = self._create_temporal_bucket()
            files_uri = self._export_result_to_storage(table_ref=table_ref, destination_bucket=bucket)
            temp_folder, files_regex = self._download_files_from_storage(files_uri=files_uri)
            loaded_data = h2o.import_file(path=temp_folder+"/",pattern = ".*\.csv.gz")
            try:
                self._remove_temporal_bucket(bucket)
                self._remove_temp_folder(path=temp_folder)
                dataset_ref = self.bigquery_client.dataset(destination_dataset)
                dataset = bigquery.Dataset(dataset_ref)
                if delete_dataset:
                    time.sleep(30) #just in case the export didn't finished yet
                    self.bigquery_client.delete_dataset(dataset, delete_contents=True)
            except Exception as e:
                logging.error(e)
            return loaded_data

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
        extract_job.done()
        time.sleep(15)                                
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
        temporal_bucket_name = "temp_spike_{0}".format(uuid.uuid1(10)).replace("-","_")
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
    
    def _convert_h2o_time_to_date(self, dataframe):
        time_cols = dataframe.columns_by_type(coltype="time")
        for col in time_cols:
            d = dataframe[int(col)].as_data_frame()/1000
            col_name = d.columns[0]
            d[col_name] = d[[col_name]].apply(lambda x: datetime.utcfromtimestamp(x).strftime("%Y-%m-%d"), axis = 1)
            dataframe[col_name] = h2o.H2OFrame(d, column_types=["string"])
        return dataframe
    
    def _export_frame_to_temp_folder(self, dataframe):
        export_file_name = dataframe.frame_id+".csv"
        temp_folder = tempfile.mkdtemp()
        export_file_path = "{0}/{1}".format(temp_folder, export_file_name)
        #TODO: Matias Aravena, permitir exportar en multiples archivos.
        h2o.export_file(dataframe, path=export_file_path, force=True)
        return export_file_path, temp_folder
    
    def _upload_file_to_storage(self, file_path=None, storage_path=None):
        temporal_bucket = None
        file_name = file_path.split("/")[-1]
        if file_path == None:
            raise ValueError("file_path is None")
        if storage_path == None:
            temporal_bucket = self._create_temporal_bucket()
            storage_path = "gs://{0}/".format(temporal_bucket.name)
        try:
            subprocess.check_call(['gsutil','-q','cp', file_path, storage_path], stderr=sys.stdout)
            return temporal_bucket, storage_path+file_name
        except Exception as e:
            if temporal_bucket != None:
                self._remove_temporal_bucket(temporal_bucket)
            raise RuntimeError("Couldn't upload the file {0} into Storage".format(file_path))

    def _create_table_from_storage(self, storage_file_uris=None, 
                                         destination_dataset=None,
                                         destination_table=None,
                                         schema=None,
                                         append=False):
        upload_job_config = bigquery.LoadJobConfig()
        upload_job_config.skip_leading_rows = 1
        upload_job_config.location = "US"
        upload_job_config.create_disposition = "CREATE_IF_NEEDED"
        if schema == None:
            upload_job_config.autodetect = True
        else:
            schema_ = []
            for field in schema:
                schema_.append(bigquery.SchemaField(name=field['name'],
                                                    field_type=field['type'],
                                                    mode="NULLABLE"))
            upload_job_config.schema = schema_
        upload_job_config.source_format = "CSV"
        if append == True:
            upload_job_config.write_disposition = "WRITE_APPEND"
        else:
            upload_job_config.write_disposition = "WRITE_TRUNCATE"
        table_ref = "{project}.{dataset}.{table}".format(project=self.project_id,
                                                         dataset=destination_dataset,
                                                         table=destination_table)
        table_ref = bigquery.TableReference.from_string(table_ref)
        load_job = bigquery.LoadJob(job_id=str(uuid.uuid1(10)),
                                    source_uris=storage_file_uris,
                                    destination=table_ref,
                                    client=self.bigquery_client,
                                    job_config=upload_job_config)
        load_job.result()

    def _infer_bigquery_schema(self, dataframe, col_order):
        col_types = dataframe.types
        bq_col_types = []
        for column in col_order:
            schema_type = "STRING" #default
            if col_types[column] == "int":
                schema_type = "INTEGER"
            elif col_types[column] == "real":
                schema_type = "FLOAT"
            elif col_types[column] == "time":
                schema_type = "DATE"
            else:
                schema_type == "STRING"
            bq_col_types.append({"name":column, "type":schema_type})
        return bq_col_types

    def upload_frame_to_bigquery(self, dataframe=None, destination_dataset=None, destination_table=None, append=True):
        """Loads data from BigQuery directly into H2O.
        Arguments:
            dataframe: H2OFrame with the data.
            destination_dataset: str destination dataset used.
            destination_table: str destination table used.
        """
        #dataframe_copy = h2o.deep_copy(dataframe, idx=uuid.uuid1(10))
        #TODO: check if data_frame copy use too much memory for big datasets
        if (destination_dataset == None or destination_table==None):
            raise ValueError("No destination_dataset or destination_table provided")
        try:
            original_col_order = dataframe.col_names
            bq_schema = self._infer_bigquery_schema(dataframe, col_order=original_col_order)
            dataframe = self._convert_h2o_time_to_date(dataframe)
            exported_file, temp_folder = self._export_frame_to_temp_folder(dataframe)
            tmp_bucket, storage_file_path = self._upload_file_to_storage(file_path=exported_file)

            self._create_table_from_storage(storage_file_uris=storage_file_path, 
                                            destination_dataset=destination_dataset,
                                            destination_table=destination_table,
                                            schema=bq_schema,
                                            append=append)
            self._remove_temp_folder(temp_folder)
            if tmp_bucket != None:
                self._remove_temporal_bucket(tmp_bucket)
        except Exception as e:
            self._remove_temp_folder(temp_folder)
            if tmp_bucket != None:
                self._remove_temporal_bucket(tmp_bucket)