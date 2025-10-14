# talent_scout/data_collection/fbref_scraper.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data_ingest.config import DATABASE_URL
from data_ingest.db import SessionLocal, Base
from data_ingest.models import Player, PlayerStats

logger = logging.getLogger(__name__)

class FBrefSeleniumScraperCorrected:
    def __init__(self, delay: float = 5.0):
        self.delay = delay
        self.base_url = "https://fbref.com"
        self.driver = None
        self.setup_driver()
    
    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        print("✅ Driver Chrome configuré")
    
    def scrape_top_leagues(self, seasons: List[str] = None) -> pd.DataFrame:
        if seasons is None:
            seasons = ["2024-2025"]
        
        leagues = {
            'Premier League': '9',
            'La Liga': '12', 
            'Serie A': '11',
            'Bundesliga': '20',
            'Ligue 1': '13'
        }
        
        all_players = []
        for league_name, league_id in leagues.items():
            logger.info(f"🔍 Scraping {league_name}...")
            for season in seasons:
                try:
                    players_data = self._scrape_league_season(league_id, league_name, season)
                    all_players.extend(players_data)
                    time.sleep(self.delay)
                except Exception as e:
                    logger.error(f"❌ Erreur {league_name} {season}: {e}")
                    continue
        return pd.DataFrame(all_players)
    
    def _scrape_league_season(self, league_id: str, league_name: str, season: str) -> List[Dict]:
        url = f"{self.base_url}/en/comps/{league_id}/{season}/stats/{season}-Stats"
        try:
            print(f"🌐 Navigation vers: {url}")
            self.driver.get(url)
            wait = WebDriverWait(self.driver, 20)
            wait.until(EC.presence_of_element_located((By.ID, "stats_standard")))
            print("✅ Page chargée avec succès")
            self.driver.execute_script("window.scrollTo(0, 500);")
            time.sleep(2)
            players_data = self._parse_standard_stats_corrected(league_name, season)
            logger.info(f"✅ {league_name} {season}: {len(players_data)} joueurs")
            return players_data
        except Exception as e:
            logger.error(f"❌ Erreur scraping {league_name} {season}: {e}")
            return []
    
    def _parse_standard_stats_corrected(self, league_name: str, season: str) -> List[Dict]:
        players = []
        try:
            table = self.driver.find_element(By.ID, "stats_standard")
            rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
            print(f"🔍 {len(rows)} lignes trouvées")
            successful = 0
            for i, row in enumerate(rows):
                if i % 50 == 0:
                    print(f"   📝 Traitement ligne {i}/{len(rows)}...")
                try:
                    player_data = self._extract_player_data_corrected(row, league_name, season, i)
                    if player_data:
                        players.append(player_data)
                        successful += 1
                        if successful <= 3:
                            print(f"      ✅ {player_data['player_name']} - {player_data['team']} ({player_data['goals']} buts)")
                except Exception as e:
                    continue
            print(f"   🎯 {successful} joueurs extraits avec succès")
            return players
        except Exception as e:
            logger.error(f"❌ Erreur extraction: {e}")
            return []
    
    def _extract_player_data_corrected(self, row, league_name: str, season: str, i: int) -> Optional[Dict]:
        try:
            cells = row.find_elements(By.XPATH, ".//th | .//td")

            if len(cells) < 10:
                return None

            # ✅ Extraction des infos principales
            player_name = cells[1].text.strip()
            team = cells[4].text.strip()
            position = cells[3].text.strip()
            age = self._parse_age(cells[5].text)
            minutes_played = self._parse_int(cells[9].text)
            goals = self._parse_int(cells[11].text)
            assists = self._parse_int(cells[12].text)
            nationality = cells[2].text.strip()

            fbref_id = None
            try:
                player_link = cells[1].find_element(By.XPATH, ".//a[contains(@href, '/players/')]")
                if player_link:
                    player_url = player_link.get_attribute('href')
                    parts = player_url.split('/')
                    if len(parts) >= 6:
                        fbref_id = parts[5]
            except Exception:
                pass

            stats = {
                'fbref_id': fbref_id,
                'player_name': player_name,
                'position': position,
                'team': team,
                'age': age,
                'nationality': nationality, 
                'minutes_played': minutes_played,
                'goals': goals,
                'assists': assists,
                'goals_per90': 0,
                'assists_per90': 0,
                'league': league_name,
                'season': season,
                'scraped_at': datetime.now(timezone.utc)
            }

            if minutes_played > 0:
                stats['goals_per90'] = round((goals / minutes_played) * 90, 2)
                stats['assists_per90'] = round((assists / minutes_played) * 90, 2)

            return stats if minutes_played >= 90 else None

        except Exception as e:
            print(f"⚠️ Erreur ligne {i}: {e}")
            return None
    
    def _parse_int(self, text: str) -> int:
        try:
            clean_text = text.strip().replace(',', '')
            return int(clean_text) if clean_text else 0
        except:
            return 0

    def _parse_age(self, text: str) -> int:
        try:
            return int(text.split('-')[0]) if '-' in text else int(text)
        except:
            return None

    def save_to_database(self, players_df: pd.DataFrame):
        """Sauvegarder les joueurs dans la base de données - SOLUTION SIMPLIFIÉE"""
        session = SessionLocal()
        try:
            # 🔥 ÉTAPE 0 : CRÉER LES TABLES SI ELLES N'EXISTENT PAS
            print("🔨 Création des tables si nécessaire...")
            from data_ingest.db import Base, engine
            Base.metadata.create_all(bind=engine)  # Cette ligne crée les tables manquantes
            print("✅ Tables vérifiées/créées")
            # 🔥 ÉTAPE 1 : Supprimer les doublons du DataFrame
            print("🧹 Suppression des doublons...")
            players_df_clean = players_df.drop_duplicates(subset=['fbref_id'], keep='first')
            print(f"📊 Après nettoyage : {len(players_df_clean)} joueurs uniques (sur {len(players_df)} originaux)")
            
            # 🔥 ÉTAPE 2 : Vider les tables existantes (pour repartir à zéro)
            print("🗑️  Nettoyage des tables existantes...")
            session.query(PlayerStats).delete()
            session.query(Player).delete()
            session.commit()
            
            # 🔥 ÉTAPE 3 : Insertion simple des joueurs
            print("💾 Insertion des joueurs...")
            for _, player_data in players_df_clean.iterrows():
                clean_data = {}
                for key, value in player_data.items():
                    if hasattr(value, 'isoformat'):
                        clean_data[key] = value.isoformat()
                    else:
                        clean_data[key] = value
                
                # Créer le joueur
                player = Player(
                    fbref_id=clean_data.get('fbref_id'),
                    name=clean_data['player_name'],
                    position=clean_data['position'],
                    team=clean_data['team'],
                    nationality=clean_data['nationality'],
                    age=clean_data['age'],
                    data=clean_data
                )
                session.add(player)
            
            session.commit()
            print(f"✅ {len(players_df_clean)} joueurs insérés")
            
            # 🔥 ÉTAPE 4 : Ajouter les statistiques
            print("📈 Ajout des statistiques...")
            for _, player_data in players_df_clean.iterrows():
                clean_data = {}
                for key, value in player_data.items():
                    if hasattr(value, 'isoformat'):
                        clean_data[key] = value.isoformat()
                    else:
                        clean_data[key] = value
                
                fbref_id = clean_data.get('fbref_id')
                if fbref_id:
                    # Trouver le joueur par fbref_id
                    player = session.query(Player).filter_by(fbref_id=fbref_id).first()
                    
                    if player:
                        stats = PlayerStats(
                            player_id=player.id,
                            season=clean_data['season'],
                            competition=clean_data['league'],
                            minutes_played=clean_data['minutes_played'],
                            goals=clean_data['goals'],
                            assists=clean_data['assists'],
                            data=clean_data
                        )
                        session.add(stats)
            
            session.commit()
            print(f"📊 {len(players_df_clean)} statistiques ajoutées")
            
            logger.info(f"💾 Sauvegarde terminée : {len(players_df_clean)} joueurs uniques")
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Erreur sauvegarde base: {e}")
            import traceback
            traceback.print_exc()
        finally:
            session.close()

    def close(self):
        if self.driver:
            self.driver.quit()
            print("🔚 Driver fermé")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = None
    try:
        scraper = FBrefSeleniumScraperCorrected(delay=5.0)
        print("🚀 Début du scraping FBref avec debug structure...")
        players_df = scraper.scrape_top_leagues(seasons=["2024-2025"])
        print(f"✅ {len(players_df)} joueurs scrapés")
        if len(players_df) > 0:
            # Sauvegarde en base
            scraper.save_to_database(players_df)
            
            # Sauvegarde CSV
            players_df.to_csv("data/players_stats_complete.csv", index=False)
            
            # Statistiques détaillées
            print(f"\n🎯 STATISTIQUES COMPLÈTES:")
            print(f"   • Total joueurs: {len(players_df)}")
            print(f"   • Équipes uniques: {players_df['team'].nunique()}")
            print(f"   • Buts totaux: {players_df['goals'].sum()}")
            print(f"   • Passes décisives: {players_df['assists'].sum()}")
            print(f"   • Minutes totales: {players_df['minutes_played'].sum():,}")
            
            print(f"\n🏆 TOP 10 BUTEURS:")
            top_scorers = players_df.nlargest(10, 'goals')[['player_name', 'team', 'league', 'goals', 'assists']]
            print(top_scorers.to_string(index=False))
            
            print(f"\n🎯 TOP 10 PASSEURS:")
            top_assisters = players_df.nlargest(10, 'assists')[['player_name', 'team', 'league', 'goals', 'assists']]
            print(top_assisters.to_string(index=False))

    except Exception as e:
        print(f"💥 ERREUR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if scraper:
            scraper.close()
    print("\n" + "=" * 60)
