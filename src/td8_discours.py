import sys
from pathlib import Path

# Astuce : si ton code est dans src/, ça permet d'importer correctement
sys.path.append("src")

import re
import inspect
from datetime import datetime
import pandas as pd

from Corpus import Corpus
from Document import Document
from SearchEngine import SearchEngine


# Utils texte / parsing
def split_into_sentences(text: str) -> list[str]:
    """
    Découpe basique en phrases.
    On considère qu'une phrase se termine par . ! ?
    """
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    s = re.sub(r"\s+", " ", s)
    parts = re.split(r"(?<=[.!?])\s+", s)
    return [p.strip() for p in parts if p and p.strip()]


def strip_outer_quotes(x) -> str:
    """
    Enlève les guillemets externes "..." si présents.
    """
    if x is None:
        return ""
    s = str(x).strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def parse_date_safe(x):
    """
    Parse une date de façon "tolérante".
    Supporte :
    - "April 12, 2015"
    - "2016-09-30"
    - ISO8601 (avec timezone)
    Retour :
    - datetime ou None
    """
    if x is None:
        return None

    s = strip_outer_quotes(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None

    # Format texte (ex: April 12, 2015)
    try:
        return datetime.strptime(s, "%B %d, %Y")
    except Exception:
        pass

    # ISO etc.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


# Lecture robuste d'un TSV "cassé" (multi-ligne)
def read_discours_tsv(csv_path: Path) -> pd.DataFrame:
    """
    TSV (tabulations) avec parfois du texte multi-ligne dans la colonne "text".
    Du coup, une lecture pd.read_csv classique peut "sauter" des lignes.

    Stratégie ici :
    - on lit ligne par ligne
    - on accumule dans un buffer
    - dès qu'on arrive à reconstituer 5 champs (speaker, text, date, descr, link),
      on valide l'enregistrement.
    """
    print("Séparateur forcé : '\\t' (TSV) + lecture robuste (anti-skip)")

    rows = []
    buffer = ""

    def try_parse(buf: str):
        parts = buf.rstrip("\n").split("\t", 4)  # max 5 colonnes
        return parts if len(parts) == 5 else None

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            buffer = (buffer + raw) if buffer else raw

            parsed = try_parse(buffer)
            if parsed is None:
                continue

            speaker, text, date, descr, link = parsed

            # Skip du header si présent
            sp = speaker.strip().strip('"').lower()
            dt = date.strip().strip('"').lower()
            lk = link.strip().strip('"').lower()
            if sp == "speaker" and dt == "date" and lk == "link":
                buffer = ""
                continue

            rows.append({
                "speaker": strip_outer_quotes(speaker.strip()),
                "text": strip_outer_quotes(text.strip()),
                "date": strip_outer_quotes(date.strip()),
                "descr": strip_outer_quotes(descr.strip()),
                "link": strip_outer_quotes(link.strip()),
            })
            buffer = ""

    return pd.DataFrame(rows)


# TD6 : checks Corpus.search / Corpus.concorde
def safe_call(method, *args, **kwargs):
    """
    Appelle une méthode en gérant les erreurs de signature.
    """
    try:
        return method(*args, **kwargs)
    except TypeError as e:
        sig = None
        try:
            sig = inspect.signature(method)
        except Exception:
            pass
        print(f"Appel impossible: {method.__name__}{sig if sig else ''}")
        print(f"Détail: {e}")
        return None


def run_td6_checks(corpus: Corpus):
    """
    Petit check rapide sur search/concorde (si présents).
    Le but : vérifier que tes méthodes TD6 fonctionnent toujours
    même sur le nouveau corpus (discours).
    """
    print("\n TD6 checks: Corpus.search & Corpus.concorde ---")

    # search
    if hasattr(corpus, "search"):
        print("corpus.search() trouvé")
        out = safe_call(corpus.search, "climate")
        if out is None:
            out = safe_call(corpus.search, "climate", 10)
        if out is None:
            out = safe_call(corpus.search, "climate", top_k=10)

        if out is not None:
            print("Aperçu corpus.search('climate') :")
            try:
                print(out.head(5) if hasattr(out, "head") else str(out)[:800])
            except Exception:
                print(str(out)[:800])
    else:
        print("corpus.search() introuvable.")

    # concorde
    if hasattr(corpus, "concorde"):
        print("corpus.concorde() trouvé")
        out = safe_call(corpus.concorde, "climate", 30)
        if out is None:
            out = safe_call(corpus.concorde, "climate", 30, 10)
        if out is None:
            out = safe_call(corpus.concorde, "climate", context=30)

        if out is not None:
            print("Aperçu corpus.concorde('climate', ...) :")
            try:
                print(out.head(5) if hasattr(out, "head") else str(out)[:800])
            except Exception:
                print(str(out)[:800])
    else:
        print("corpus.concorde() introuvable.")


# Construction Corpus + SearchEngine
def build_corpus_from_df(df: pd.DataFrame) -> tuple[Corpus, int]:
    """
    Construit un corpus où 1 phrase = 1 document.
    Retour :
    - corpus
    - nb de phrases/documents ajoutés
    """
    Corpus.reset_singleton()
    corpus = Corpus("Corpus TD8 - Discours US")

    total_sentences = 0

    for _, row in df.iterrows():
        speaker = str(row.get("speaker", "Unknown")).strip() or "Unknown"
        title = str(row.get("descr", "Discours")).strip() or "Discours"
        date = parse_date_safe(row.get("date", None))
        url = str(row.get("link", "")).strip()
        text = row.get("text", "")

        # On découpe le discours en phrases
        for sent in split_into_sentences(text):
            corpus.add_document(
                Document(
                    titre=title,
                    auteur=speaker,
                    date=date,
                    url=url,
                    texte=sent,
                )
            )
            total_sentences += 1

    return corpus, total_sentences


def build_search_engine(corpus: Corpus) -> SearchEngine:
    """
    Construit le moteur de recherche et affiche un petit aperçu du vocab.
    """
    print("\n--- Build SearchEngine TD8 ---")
    se = SearchEngine(corpus)

    print("\n--- Vocab aperçu ---")
    try:
        print(se.get_vocab_df().head(10))
    except Exception as e:
        print("Impossible d'afficher le vocab_df:", e)

    return se


def build_search_engine_from_csv(csv_path: Path | None = None):
    """
    Fonction "propre" (utile notebook TD9/TD10) :
    - lit le TSV de discours
    - construit corpus (1 phrase = 1 doc)
    - construit SearchEngine
    Retourne df, corpus, se
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent / "data" / "discours_US.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    df = read_discours_tsv(csv_path)

    print("\n--- Aperçu DF ---")
    print(df.head())

    print("\n--- Colonnes détectées ---")
    print(list(df.columns))

    if "date" in df.columns:
        print("\n--- Exemple valeurs brutes date (5) ---")
        print(df["date"].head(5).tolist())

    if "speaker" in df.columns:
        print("\n--- value_counts(speaker) ---")
        print(df["speaker"].value_counts())

    corpus, total_sentences = build_corpus_from_df(df)

    print("\n--- Corpus TD8 ---")
    print(corpus)
    print(f"Nombre de phrases/documents: {total_sentences}")

    # Checks TD6
    run_td6_checks(corpus)

    se = build_search_engine(corpus)
    return df, corpus, se


# UI Jupyter (ipywidgets)
def build_widgets_ui(se: SearchEngine):
    """
    Affiche une petite UI ipywidgets (dans un notebook).
    Si ipywidgets n'est pas installé : on affiche un message.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except Exception as e:
        print("ipywidgets/IPython non dispo. Installe via: pip install ipywidgets")
        print("Détail:", e)
        return None

    title = widgets.HTML("<h3>TD9-10 - Interface de recherche (Discours US)</h3>")

    query = widgets.Text(
        value="climate change",
        description="Query:",
        placeholder="Tape ta requête…",
        layout=widgets.Layout(width="70%")
    )

    topk = widgets.IntSlider(
        value=10,
        min=1,
        max=50,
        step=1,
        description="Top K:",
        continuous_update=False,
        readout=True
    )

    btn = widgets.Button(description="Search", button_style="success")
    out = widgets.Output()

    def on_click(_):
        with out:
            clear_output()
            q = query.value.strip()
            if not q:
                print("Requête vide.")
                return
            try:
                res = se.search(q, top_k=int(topk.value), show_progress=False)
                display(res)
            except Exception as e:
                print("Erreur pendant la recherche:", e)

    btn.on_click(on_click)

    ui = widgets.VBox([
        title,
        widgets.HBox([query, btn]),
        topk,
        out
    ])

    display(ui)
    return ui


# Main (CLI)
def main():
    """
    Exécution en ligne de commande :
    - construit df/corpus/se
    - lance quelques requêtes de test
    - si notebook détecté, propose l'UI ipywidgets
    """
    csv_path = Path(__file__).resolve().parent.parent / "data" / "discours_US.csv"

    df, corpus, se = build_search_engine_from_csv(csv_path)

    # Requêtes de test rapides
    queries = ["climate change", "immigration border", "economy jobs", "health care"]
    for q in queries:
        print(f"\n=== Query: {q} ===")
        res = se.search(q, top_k=10, show_progress=True)
        print(res)

    # Détection notebook
    if "ipykernel" in sys.modules:
        print("\nNotebook détecté -> affichage UI ipywidgets")
        build_widgets_ui(se)
    else:
        print("\nPour voir l'interface :")
        print("   - ouvre TD9_10_Interface.ipynb")
        print("   - appelle build_widgets_ui(se)")

    return df, corpus, se


if __name__ == "__main__":
    main()
