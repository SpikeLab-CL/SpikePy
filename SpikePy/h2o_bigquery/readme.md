
## BigQuery H2O Data Loader

Load data from BigQuery into H2O from querys.

#### Arguments.

```

h2o_bigquery.data_loader.H2OBigQueryLoader().from_query()

Arguments:
    query: str query to run in BigQuery.
    allow_large_results: bool If set in True, allow to download big datasets
    destination_dataset: str destination dataset used if allow_large_results is set True
                            if is set None, then a temporal dataset is created.
    destination_table: str destination table used if allow_large_results is set True
                            if is set None, then a temporal table is created.
```

```
h2o_bigquery.data_loader.H2OBigQueryLoader().upload_frame_to_bigquery()

Loads data from BigQuery directly into H2O.

Arguments:
    dataframe: H2OFrame with the data.
    destination_dataset: str destination dataset used.
    destination_table: str destination table used.
```
#### Usage example.

```
h2o.init(port=54321)
loader = H2OBigQueryLoader(project_id="gcp_project")
query = "SELECT * FROM `gcp_project.dataset.table` LIMIT 1000"
dataframe = loader.from_query(query=query, allow_large_results=True)
print(dataframe.head())
h2o.cluster().shutdown()
```

```
h2o.init()
loader = H2OBigQueryLoader(project_id="gcp_project")
data = h2o.upload_file("sample_test.csv")
loader.upload_frame_to_bigquery(dataframe=data, 
                                destination_dataset="spike_tmp",
                                destination_table="h2o_upload_table")
h2o.cluster().shutdown()
```
