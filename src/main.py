"""
Ce script sert d'orchestrateur pour exécuter le TD3 :
- Charger la configuration (variables d’environnement via .env)
- Lancer le pipeline d’acquisition (Reddit + Arxiv)
- Produire un corpus final au format TSV
- Éviter de solliciter les API à chaque exécution (cache sur disque)
"""

import os

from dotenv import load_dotenv
from td3_acquisition import run_td3


# Chargement des variables d’environnement (.env)
load_dotenv()


if __name__ == "__main__":
    # Paramètres de l’expérience (TD3)

    # Mot-clé de recherche
    keywords = "climate"

    # Chemin du fichier TSV final
    out_path = "data/corpus.tsv"

    # Identifiants Reddit (variables d’environnement)
    reddit_credentials = {
        "client_id": os.getenv("REDDIT_CLIENT_ID"),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
        # user_agent requis par Reddit API 
        "user_agent": os.getenv("REDDIT_USER_AGENT", "M1-IR-TD3"),
        # subreddit : "all" permet une recherche globale
        "subreddit": "all",
    }

    # Lancement du pipeline TD3
    df, global_str = run_td3(
        keywords=keywords,
        reddit_limit=50,  # nombre de documents Reddit à récupérer
        arxiv_limit=50,   # nombre de documents Arxiv à récupérer
        out_path=out_path,
        reddit_credentials=reddit_credentials,
    )

    # Valider rapidement que :
    # - le corpus est bien chargé
    # - la chaîne globale est bien construite
    # - la structure du DataFrame est cohérente
    print("\n--- Vérification TD3 ---")
    print(f"Nombre de documents chargés : {len(df)}")
    print(f"Taille de la chaîne globale : {len(global_str)} caractères")

    # Aperçu du DataFrame
    print("\nAperçu du corpus :")
    print(df.head())
