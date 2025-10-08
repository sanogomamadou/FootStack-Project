import requests
from urllib.parse import urljoin
from .config import API_TOKEN, API_BASE
from .utils import safe_get

HEADERS = {"X-Auth-Token": API_TOKEN} if API_TOKEN else {}

def get_competitions(plan="TIER_ONE"):
    # exemple: https://api.football-data.org/v4/competitions
    url = urljoin(API_BASE, "competitions")
    r = safe_get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def get_competition_standings(competition_id):
    url = urljoin(API_BASE, f"competitions/{competition_id}/standings")
    r = safe_get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def get_matches(competition_id, date_from=None, date_to=None):
    url = urljoin(API_BASE, f"competitions/{competition_id}/matches")
    params = {}
    if date_from: params["dateFrom"] = date_from
    if date_to: params["dateTo"] = date_to
    r = safe_get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()
