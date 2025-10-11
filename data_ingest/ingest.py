# data_ingest/ingest.py
import argparse
from datetime import datetime, timedelta, timezone
from sqlalchemy.exc import IntegrityError

from .db import engine, SessionLocal, Base
from .models import Competition, Team, Match
from .api_client import get_competitions, get_matches, get_competition_standings

def init_db():
    print("Création des tables si besoin...")
    Base.metadata.create_all(bind=engine)
    print("DB prête.")

def ingest_competitions(session):
    data = get_competitions()
    count = 0
    for comp in data.get("competitions", []):
        obj = Competition(
            id=comp.get("id"),
            name=comp.get("name"),
            area_name=comp.get("area", {}).get("name"),
            code=comp.get("code"),
            data=comp
        )
        session.merge(obj)
        count += 1
    session.commit()
    print(f"Ingested {count} competitions.")

def ingest_matches_for_competition(session, competition_id, days_back=365, date_from=None, date_to=None):
    # Calcul des dates si non fournies
    if date_from and date_to:
        date_from_dt = datetime.strptime(date_from, "%Y-%m-%d").date()
        date_to_dt = datetime.strptime(date_to, "%Y-%m-%d").date()
    else:
        date_to_dt = datetime.now(timezone.utc).date()
        date_from_dt = date_to_dt - timedelta(days=days_back)

    raw = get_matches(competition_id, date_from=str(date_from_dt), date_to=str(date_to_dt))
    matches = raw.get("matches", [])
    print(f"Found {len(matches)} matches for competition {competition_id} from {date_from_dt} to {date_to_dt}")
    
    for m in matches:
        try:
            match_obj = Match(
                id=m.get("id"),
                competition_id=competition_id,
                utc_date=m.get("utcDate"),
                status=m.get("status"),
                matchday=m.get("matchday"),
                home_team_id=m.get("homeTeam", {}).get("id"),
                away_team_id=m.get("awayTeam", {}).get("id"),
                score=m.get("score"),
                raw=m
            )
            session.merge(match_obj)
        except IntegrityError:
            session.rollback()
    session.commit()
    print("Matches persisted.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-db", action="store_true")
    parser.add_argument("--competitions", action="store_true")
    parser.add_argument("--ingest-matches", type=int, help="comp id pour ingester les matches")
    parser.add_argument("--days-back", type=int, default=365)
    parser.add_argument("--date-from", type=str, help="YYYY-MM-DD")
    parser.add_argument("--date-to", type=str, help="YYYY-MM-DD")
    args = parser.parse_args()

    init_db()
    session = SessionLocal()
    try:
        if args.competitions:
            ingest_competitions(session)
        if args.ingest_matches:
            ingest_matches_for_competition(
                session, 
                args.ingest_matches, 
                days_back=args.days_back,
                date_from=args.date_from,
                date_to=args.date_to
            )
    finally:
        session.close()

if __name__ == "__main__":
    main()
