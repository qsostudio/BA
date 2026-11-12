import ee, json
from google.oauth2 import service_account as google_sa

with open("gee_key.json") as f:
    key_data = json.load(f)
scopes = ["https://www.googleapis.com/auth/earthengine"]
creds = google_sa.Credentials.from_service_account_info(key_data, scopes=scopes)
ee.Initialize(creds)
print("✅ GEE inicializado correctamente")
