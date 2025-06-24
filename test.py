from google.cloud import bigquery
from dotenv import load_dotenv
load_dotenv()
client = bigquery.Client()
print("Test passed:", [r.ok for r in client.query("SELECT 1 AS ok").result()][0])
