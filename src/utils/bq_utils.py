from google.cloud import bigquery
from io import StringIO
import json

def load_json_to_bq(feedback_dict, schema,output_table_name):
    bigquery_client = bigquery.Client()
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = False
    job_config.write_disposition = "WRITE_APPEND"
    job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON

    # ref_table = bigquery_client.get_table(output_table_name)
    job_config.schema = schema

    # set buffer
    buffer = StringIO()
    json.dump(feedback_dict, buffer)
    buffer.seek(0) 

    # write to big query
    print(f"Appending {feedback_dict} to {output_table_name}")
    job = bigquery_client.load_table_from_file(buffer,output_table_name,job_config=job_config)
    result = job.result()
    bigquery_client.close()
    return result