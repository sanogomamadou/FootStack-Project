# data_ingest/config.py
from dotenv import load_dotenv
import os

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
API_BASE = "https://api.football-data.org/v4/"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/footstack")

# Timeout & rate-limit conservative defaults
REQUEST_TIMEOUT = 10
RATE_LIMIT_SLEEP = 1.2  # seconds between calls (ajuste si l'API est stricte)
