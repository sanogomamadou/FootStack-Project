# talent_scout/data_collection/data_processor.py
import pandas as pd
from sqlalchemy.orm import Session
from data_ingest.models import Player, PlayerStats
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data_ingest.config import DATABASE_URL
from data_ingest.db import SessionLocal, Base

def process_player_data(save_csv=True):
    session = SessionLocal()
    try:
        # Charger les données
        players = pd.read_sql(session.query(Player).statement, session.bind)
        stats = pd.read_sql(session.query(PlayerStats).statement, session.bind)

        print(f"  {len(players)} joueurs | {len(stats)} stats chargées")

        #  Fusionner les tables
        merged = pd.merge(
            stats,
            players,
            left_on='player_id',
            right_on='id',
            suffixes=('_stats', '_info')
        )

        # Feature Engineering
        merged["goal_contrib_per90"] = (merged["goals"] + merged["assists"]) / (merged["minutes_played"] / 90)
        merged["minutes_played_norm"] = merged["minutes_played"] / merged["minutes_played"].max()

        # Normalisation légère (évite les valeurs extrêmes)
        merged["goals_per90"] = merged["goals"] / (merged["minutes_played"] / 90)
        merged["assists_per90"] = merged["assists"] / (merged["minutes_played"] / 90)

        # Nettoyage
        merged = merged.fillna(0)
        merged = merged[merged["minutes_played"] >= 300]  # Écarte les joueurs peu utilisés

        # Regrouper les positions
        def map_position(pos):
            if pos is None:
                return "Unknown"
            if any(x in pos for x in ["GK", "Goalkeeper"]):
                return "Goalkeeper"
            elif any(x in pos for x in ["DF", "Back", "Defender"]):
                return "Defender"
            elif any(x in pos for x in ["MF", "Midfield"]):
                return "Midfielder"
            elif any(x in pos for x in ["FW", "Winger", "Striker", "Forward"]):
                return "Forward"
            return "Other"

        merged["position_group"] = merged["position"].apply(map_position)

        merged.drop(columns=["data_stats", "data_info"], inplace=True)

        merged["nationality"] = merged["nationality"].apply(lambda x: ' '.join(x.split()[1:]) if isinstance(x, str) else x)


        print("  Transformation terminée")

        # Sauvegarde locale pour inspection
        if save_csv:
            os.makedirs("data", exist_ok=True)
            merged.to_csv("data/processed_players.csv", index=False)
            print("  Fichier sauvegardé : data/processed_players.csv")

        return merged

    finally:
        session.close()

if __name__ == "__main__":
    df = process_player_data()
    print(df.head())
