from __future__ import annotations

import os
import re
from typing import List, Dict, Tuple, Optional
from urllib.parse import quote
from urllib.request import urlopen
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xmltodict


# Charger .env automatiquement
def _load_env_if_possible() -> None:
    """
    Charge le fichier .env à la racine du projet (si python-dotenv est installé).
    Ça évite le cas "j'ai un .env mais Python ne le lit pas".
    """
    try:
        from dotenv import load_dotenv  # pip install python-dotenv
        project_root = Path(__file__).resolve().parent.parent
        load_dotenv(project_root / ".env")
    except Exception:
        # Pas grave les variables peuvent déjà être dans l'environnement
        pass


# Helpers – Nettoyage et conversion
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_iso_from_utc_ts(ts: Optional[float]) -> str:
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return ""


# Partie 1.1 – Reddit (API PRAW)
def fetch_reddit_docs(
    keywords: str,
    limit: int,
    client_id: str,
    client_secret: str,
    user_agent: str,
    subreddit: str = "all",
) -> List[Dict[str, object]]:
    import praw

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    docs: List[Dict[str, object]] = []

    # Important : forcer un mode "safe" pour éviter des erreurs silencieuses
    try:
        iterator = reddit.subreddit(subreddit).search(keywords, limit=limit)
    except Exception as e:
        print(" Erreur Reddit (initialisation search) :", e)
        return []

    for submission in iterator:
        title = normalize_text(getattr(submission, "title", "") or "")
        body = normalize_text(getattr(submission, "selftext", "") or "")

        full_text = (title + " " + body).strip()
        if not full_text:
            continue

        author_obj = getattr(submission, "author", None)
        author_name = "Unknown"
        try:
            if author_obj is not None and getattr(author_obj, "name", None):
                author_name = str(author_obj.name)
        except Exception:
            author_name = "Unknown"

        created_utc = getattr(submission, "created_utc", None)
        date_iso = to_iso_from_utc_ts(created_utc)

        url = normalize_text(getattr(submission, "url", "") or "")
        nb_comments = getattr(submission, "num_comments", 0) or 0

        docs.append(
            {
                "origine": "reddit",
                "titre": title if title else "Reddit post",
                "auteur": author_name,
                "date": date_iso,
                "url": url,
                "texte": full_text,
                "nb_commentaires": int(nb_comments),
                "co_auteurs": "",
            }
        )

    return docs


# Partie 1.2 – Arxiv (API ATOM)
def fetch_arxiv_docs(keywords: str, limit: int) -> List[Dict[str, object]]:
    query = quote(keywords)
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={limit}"

    with urlopen(url) as resp:
        xml_bytes = resp.read()

    data = xmltodict.parse(xml_bytes)
    feed = data.get("feed", {})
    entries = feed.get("entry", [])

    if isinstance(entries, dict):
        entries = [entries]

    docs: List[Dict[str, object]] = []

    for entry in entries:
        title = normalize_text(entry.get("title", "") or "")
        summary = normalize_text(entry.get("summary", "") or "")
        full_text = (title + " " + summary).strip()
        if not full_text:
            continue

        arxiv_url = normalize_text(entry.get("id", "") or "")
        published = normalize_text(entry.get("published", "") or "")

        authors_raw = entry.get("author", [])
        authors_list: List[str] = []

        if isinstance(authors_raw, dict):
            name = authors_raw.get("name", "")
            if name:
                authors_list = [normalize_text(name)]
        elif isinstance(authors_raw, list):
            for a in authors_raw:
                if isinstance(a, dict) and a.get("name"):
                    authors_list.append(normalize_text(a["name"]))

        main_author = authors_list[0] if authors_list else "Unknown"
        co_authors = authors_list[1:] if len(authors_list) > 1 else []

        docs.append(
            {
                "origine": "arxiv",
                "titre": title if title else "Arxiv paper",
                "auteur": main_author,
                "date": published,
                "url": arxiv_url,
                "texte": full_text,
                "nb_commentaires": 0,
                "co_auteurs": ";".join(co_authors),
            }
        )

    return docs


# Partie 2 – DataFrame + persistance (TSV)
def build_dataframe(
    reddit_docs: List[Dict[str, object]],
    arxiv_docs: List[Dict[str, object]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    idx = 0

    for d in reddit_docs + arxiv_docs:
        row = dict(d)
        row["id"] = idx
        rows.append(row)
        idx += 1

    cols = [
        "id", "origine", "titre", "auteur", "date",
        "url", "texte", "nb_commentaires", "co_auteurs"
    ]
    return pd.DataFrame(rows, columns=cols)


def _resolve_out_path(out_path: str) -> Path:
    p = Path(out_path)
    project_root = Path(__file__).resolve().parent.parent
    if not p.is_absolute():
        p = project_root / p
    return p


def save_tsv(df: pd.DataFrame, path: str) -> Path:
    p = _resolve_out_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, sep="\t", index=False)
    return p


def load_tsv(path: str) -> pd.DataFrame:
    p = _resolve_out_path(path)
    return pd.read_csv(p, sep="\t")


# Partie 3 – Statistiques + filtrage
def corpus_stats_and_filter(df: pd.DataFrame) -> None:
    print(f"Taille du corpus (nb docs) = {len(df)}")

    before = len(df)
    df.drop(df[df["texte"].astype(str).str.len() < 100].index, inplace=True)
    after = len(df)

    print(f"Suppression docs < 100 caractères : {before} -> {after}")

    df.reset_index(drop=True, inplace=True)
    df["id"] = range(len(df))


def build_global_string(df: pd.DataFrame) -> str:
    return " ".join(df["texte"].astype(str).tolist())


# Orchestrateur TD3
def run_td3(
    keywords: str,
    reddit_limit: int,
    arxiv_limit: int,
    out_path: str,
    reddit_credentials: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, str]:
    out_p = _resolve_out_path(out_path)

    if out_p.exists():
        print(f"TSV trouvé : chargement depuis disque -> {out_p}")
        df = load_tsv(str(out_p))
        global_str = build_global_string(df)
        print(f"Taille chaîne globale (caractères) = {len(global_str)}")
        return df, global_str

    print("Pas de TSV : récupération via APIs (Reddit + Arxiv).")
    reddit_credentials = reddit_credentials or {}

    use_reddit = (
        reddit_limit > 0
        and reddit_credentials.get("client_id")
        and reddit_credentials.get("client_secret")
        and reddit_credentials.get("user_agent")
    )

    if use_reddit:
        print("Reddit ACTIVÉ")
        reddit_docs = fetch_reddit_docs(
            keywords=keywords,
            limit=reddit_limit,
            client_id=str(reddit_credentials.get("client_id")),
            client_secret=str(reddit_credentials.get("client_secret")),
            user_agent=str(reddit_credentials.get("user_agent")),
            subreddit=str(reddit_credentials.get("subreddit", "all")),
        )
        print(f"Reddit docs récupérés : {len(reddit_docs)}")
    else:
        reddit_docs = []
        print("Reddit désactivé : reddit_limit=0 ou identifiants manquants")
        print(f"   reddit_limit={reddit_limit}")
        print(f"   client_id présent ? {bool(reddit_credentials.get('client_id'))}")
        print(f"   client_secret présent ? {bool(reddit_credentials.get('client_secret'))}")
        print(f"   user_agent présent ? {bool(reddit_credentials.get('user_agent'))}")

    arxiv_docs = fetch_arxiv_docs(keywords=keywords, limit=arxiv_limit) if arxiv_limit > 0 else []
    print(f"Arxiv docs récupérés : {len(arxiv_docs)}")

    df = build_dataframe(reddit_docs, arxiv_docs)

    corpus_stats_and_filter(df)
    saved_path = save_tsv(df, str(out_p))
    print(f"Sauvegardé (après filtre) : {saved_path}")

    global_str = build_global_string(df)
    print(f"Taille chaîne globale (caractères) = {len(global_str)}")

    return df, global_str


# Main (pour générer data/corpus.tsv)
def main():
    _load_env_if_possible()

    out_path = "data/corpus.tsv"
    keywords = "climate change"

    reddit_limit = 50
    arxiv_limit = 50

    reddit_credentials = {
        "client_id": os.getenv("REDDIT_CLIENT_ID", "").strip(),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET", "").strip(),
        "user_agent": os.getenv("REDDIT_USER_AGENT", "").strip(),
        "subreddit": os.getenv("REDDIT_SUBREDDIT", "all").strip(),
    }

    df, global_str = run_td3(
        keywords=keywords,
        reddit_limit=reddit_limit,
        arxiv_limit=arxiv_limit,
        out_path=out_path,
        reddit_credentials=reddit_credentials,
    )

    print("\n--- OK TD3 ---")
    print(df.head(3))
    print(f"Fichier attendu : {_resolve_out_path(out_path)}")
    print(f"Longueur chaîne globale : {len(global_str)}")

    # mini check
    if "origine" in df.columns:
        print("\n--- value_counts(origine) ---")
        print(df["origine"].value_counts())


if __name__ == "__main__":
    main()
