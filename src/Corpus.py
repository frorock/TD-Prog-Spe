from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator, Tuple, Any
from datetime import datetime

import re
import pandas as pd

from Document import Document
from Author import Author
from Factory import DocumentFactory


def parse_date_safe(date_str: str) -> Optional[datetime]:
    """
    Convertit une date string en datetime, sans faire planter le programme.

    Pourquoi :
    - en TSV, on peut avoir des dates vides ou mal formatées
    - Reddit / Arxiv peuvent avoir des formats différents

    Retour :
    - datetime si OK
    - None sinon
    """
    if not date_str:
        return None

    s = str(date_str).strip()
    if not s:
        return None

    # Cas "Z" (UTC)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


@dataclass
class Corpus:
    """
    Corpus (TD4 -> TD7) :
    - stocke tous les documents (dict doc_id -> Document)
    - stocke tous les auteurs (dict nom -> Author)
    - propose des méthodes utiles : tri, stats, search regex, vocab, tokenize...
    - Singleton : pour n'avoir qu'un seul corpus global (TD5)
    - cache _global_str : évite de reconstruire la concaténation à chaque recherche (TD6)
    """

    nom: str
    documents: Dict[int, Document] = field(default_factory=dict)
    authors: Dict[str, Author] = field(default_factory=dict)
    _next_id: int = 0

    # Cache : concaténation de tous les textes (utile pour search/concorde)
    _global_str: Optional[str] = field(default=None, init=False, repr=False)

    # Singleton
    _instance: Optional["Corpus"] = None

    def __new__(cls, *args, **kwargs):
        """
        Singleton (TD5) :
        - si une instance existe déjà, on la réutilise
        - sinon on en crée une
        """
        if cls._instance is None:
            cls._instance = super(Corpus, cls).__new__(cls)
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """Utile en tests : repart d'un corpus vide."""
        cls._instance = None

    # Ajout de documents
    def add_document(self, doc: Document) -> int:
        """
        Ajoute un document au corpus et met à jour l'auteur.

        Effets :
        - doc_id auto-incrémenté
        - ajout dans self.documents
        - ajout / mise à jour de self.authors
        - invalide le cache _global_str
        """
        doc_id = self._next_id
        self.documents[doc_id] = doc
        self._next_id += 1

        author_name = doc.auteur or "Unknown"
        if author_name not in self.authors:
            self.authors[author_name] = Author(author_name)

        self.authors[author_name].add(doc_id, doc.titre)

        # le corpus a changé donc le cache n'est plus valide
        self._global_str = None

        return doc_id

    def add_document_with_id(self, doc_id: int, doc: Document) -> None:
        """
        Ajoute un document en imposant l'id (utile quand on recharge depuis TSV).
        permet de conserver la cohérence doc_id <-> position dans les matrices (TD7)
        """
        doc_id = int(doc_id)
        self.documents[doc_id] = doc
        self._next_id = max(self._next_id, doc_id + 1)

        author_name = doc.auteur or "Unknown"
        if author_name not in self.authors:
            self.authors[author_name] = Author(author_name)

        self.authors[author_name].add(doc_id, doc.titre)

        self._global_str = None

    # Parcours / accès
    def iter_docs(self) -> Iterator[Document]:
        """
        Itère sur les documents dans l'ordre des doc_id.
        C'est important pour rester stable entre :
        - corpus
        - vocab
        - matrices TF / TF-IDF
        """
        for doc_id in sorted(self.documents.keys()):
            yield self.documents[doc_id]

    def iter_docs_items(self) -> Iterator[Tuple[int, Document]]:
        """Itère sur (doc_id, doc) dans l'ordre."""
        for doc_id in sorted(self.documents.keys()):
            yield doc_id, self.documents[doc_id]

    def get_doc(self, doc_id: int) -> Optional[Document]:
        """Accès safe : pas de KeyError si l'id n'existe pas."""
        return self.documents.get(doc_id)

    def __len__(self) -> int:
        """Permet len(corpus) => nombre de documents."""
        return len(self.documents)

    # Tri / affichage
    def documents_tries_par_titre(self, n: int = 10) -> List[Document]:
        """Renvoie les n premiers documents triés par titre."""
        return sorted(self.documents.values(), key=lambda d: (d.titre or "").lower())[:n]

    def documents_tries_par_date(self, n: int = 10) -> List[Document]:
        """Renvoie les n docs les plus récents (date absente => très vieux)."""
        return sorted(
            self.documents.values(),
            key=lambda d: d.date if d.date else datetime.min,
            reverse=True,
        )[:n]

    def __repr__(self) -> str:
        """Représentation courte pour debug."""
        return f"Corpus(nom={self.nom}, nb_documents={len(self.documents)}, nb_auteurs={len(self.authors)})"

    # TD4 : sauvegarde / chargement TSV
    def save_to_tsv(self, path: str) -> None:
        """
        Sauvegarde du corpus en TSV (TD4).

        On exporte :
        - champs de base
        - + champs extra (si présents)
        """
        rows: List[Dict[str, Any]] = []

        for doc_id, doc in self.iter_docs_items():
            # origine = type logique (reddit/arxiv)
            if hasattr(doc, "getType") and callable(getattr(doc, "getType")):
                origine = str(doc.getType())
            elif hasattr(doc, "type"):
                origine = str(getattr(doc, "type"))
            else:
                origine = doc.__class__.__name__.lower()

            row: Dict[str, Any] = {
                "id": int(doc_id),
                "origine": origine,
                "titre": getattr(doc, "titre", "") or "",
                "auteur": getattr(doc, "auteur", "") or "",
                "date": getattr(doc, "date", None).isoformat() if getattr(doc, "date", None) else "",
                "url": getattr(doc, "url", "") or "",
                "texte": getattr(doc, "texte", "") or "",
            }

            # On ajoute les colonnes extra seulement si elles ne doublonnent pas les bases
            base_cols = set(row.keys())
            extra = getattr(doc, "extra", None)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if k in base_cols:
                        continue
                    # pandas peut mettre des NaN
                    if isinstance(v, float) and pd.isna(v):
                        v = ""
                    row[k] = v

            rows.append(row)

        df = pd.DataFrame(rows)

        # Colonnes principales en premier
        base_cols_order = ["id", "origine", "titre", "auteur", "date", "url", "texte"]
        other_cols = [c for c in df.columns if c not in base_cols_order]
        df = df[base_cols_order + other_cols]

        df.to_csv(path, sep="\t", index=False)

    def load_from_tsv(self, path: str) -> None:
        """
        Recharge un TSV et reconstruit le corpus.

        Important :
        - on recrée les documents via la DocumentFactory
        - on conserve les doc_id du TSV (cohérence TD7)
        """
        df = pd.read_csv(path, sep="\t")

        # Reset
        self.documents = {}
        self.authors = {}
        self._next_id = 0
        self._global_str = None

        factory = DocumentFactory()

        for _, row in df.iterrows():
            origine = str(row.get("origine", "document")).lower().strip()

            titre = str(row.get("titre", "") or "").strip()
            if not titre:
                titre = f"Document {int(row.get('id', self._next_id))}"

            auteur = str(row.get("auteur", "") or "").strip() or "Unknown"

            date_raw = str(row.get("date", "") or "").strip()
            date = parse_date_safe(date_raw)

            url = str(row.get("url", "") or "").strip()
            texte = str(row.get("texte", "") or "").strip()

            doc = factory.create(
                doc_type=origine,
                titre=titre,
                auteur=auteur,
                date=date,
                url=url,
                texte=texte,
                extra=row.to_dict(),
            )

            doc_id = int(row.get("id", self._next_id))
            self.add_document_with_id(doc_id, doc)

    # TD6 : nettoyage / search / concorde / stats
    def nettoyer_texte(self, s: str) -> str:
        """
        Nettoyage simple "fait maison" (TD6).
        - lower
        - retire chiffres
        - retire ponctuation (remplacée par espace)
        - normalise les espaces
        """
        if s is None:
            return ""

        s = str(s).lower()
        s = s.replace("\n", " ").replace("\r", " ")
        s = re.sub(r"\d+", " ", s)

        # On garde seulement les lettres + espaces (ponctuation = séparateur)
        s = re.sub(r"[^a-z\s]+", " ", s)

        s = re.sub(r"\s+", " ", s).strip()
        return s

    def iter_docs_text(self) -> List[str]:
        """Retourne la liste des textes, dans l'ordre des doc_id."""
        return [d.texte for d in self.iter_docs()]

    def build_global_string(self) -> str:
        """
        Construit la grosse chaîne concaténée.
        Optimisation : on met en cache, et on ne recalcule que si le corpus change.
        """
        if self._global_str is None:
            self._global_str = " ".join(self.iter_docs_text())
        return self._global_str

    def stats(self, n: int = 10) -> pd.DataFrame:
        """
        Stats vocab (TD6) :
        - nombre de mots différents
        - top n des mots les plus fréquents

        Retour :
        - DataFrame avec mot, tf_total, df
        """
        docs = self.iter_docs_text()

        tf: Dict[str, int] = {}
        df: Dict[str, int] = {}

        # On fait tout en une passe (plus efficace)
        for text in docs:
            tokens = self.tokenize(text)
            seen = set()
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
                if tok not in seen:
                    df[tok] = df.get(tok, 0) + 1
                    seen.add(tok)

        freq = pd.DataFrame(
            [{"mot": w, "tf_total": tf[w], "df": df.get(w, 0)} for w in tf.keys()]
        ).sort_values(by="tf_total", ascending=False, ignore_index=True)

        print("Nombre de mots différents dans le corpus :", len(tf))
        print(f"Top {n} mots les plus fréquents :")
        print(freq.head(n))

        return freq

    def search(self, motif: str, taille_contexte: int = 30, n: int = 10) -> pd.DataFrame:
        """
        Recherche dans la chaîne globale (TD6).
        - motif peut être un mot ou une regex
        - on renvoie un DataFrame avec positions + contexte gauche/droit
        """
        g = self.build_global_string()

        if not motif or not str(motif).strip():
            return pd.DataFrame(columns=["start", "end", "match", "contexte_gauche", "contexte_droit"])

        pattern = re.compile(str(motif), flags=re.IGNORECASE)
        rows: List[Dict[str, Any]] = []

        for m in pattern.finditer(g):
            start, end = m.start(), m.end()
            left = max(0, start - int(taille_contexte))
            right = min(len(g), end + int(taille_contexte))

            rows.append(
                {
                    "start": start,
                    "end": end,
                    "match": m.group(0),
                    "contexte_gauche": g[left:start],
                    "contexte_droit": g[end:right],
                }
            )

            if len(rows) >= max(0, int(n)):
                break

        return pd.DataFrame(rows)

    def concorde(self, mot: str, taille_contexte: int = 30, n: int = 10) -> pd.DataFrame:
        """
        Concordancier (TD6) :
        - recherche du mot en tant que "mot entier" 
        - renvoie contexte gauche / mot trouvé / contexte droit
        """
        g = self.build_global_string()
        target = (mot or "").strip()
        if not target:
            return pd.DataFrame(columns=["contexte_gauche", "motif_trouve", "contexte_droit"])

        safe = re.escape(target)
        pattern = re.compile(rf"\b{safe}\b", flags=re.IGNORECASE)

        rows: List[Dict[str, Any]] = []
        for m in pattern.finditer(g):
            start, end = m.start(), m.end()
            left = max(0, start - int(taille_contexte))
            right = min(len(g), end + int(taille_contexte))

            rows.append(
                {
                    "contexte_gauche": g[left:start],
                    "motif_trouve": m.group(0),
                    "contexte_droit": g[end:right],
                }
            )

            if len(rows) >= max(0, int(n)):
                break

        return pd.DataFrame(rows)

    # TD7 : tokenisation + vocab
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenisation simple et stable (TD7).
        - lower
        - retire chiffres
        - retire ponctuation (apostrophe incluse)
        - split par espaces
        """
        if not text:
            return []
        s = text.lower()
        s = re.sub(r"\d+", " ", s)
        s = re.sub(r"[^a-z\s]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s.split()

    def build_vocab(self) -> Dict[str, Dict[str, int]]:
        """
        Construit le vocabulaire (TD7).
        Sortie :
        - vocab[mot] = {id, tf_total, df}
        Contrainte :
        - mots uniques, triés alphabétiquement
        """
        tf: Dict[str, int] = {}
        df: Dict[str, int] = {}

        for text in self.iter_docs_text():
            tokens = self.tokenize(text)
            seen = set()
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
                if tok not in seen:
                    df[tok] = df.get(tok, 0) + 1
                    seen.add(tok)

        words = sorted(tf.keys())
        vocab: Dict[str, Dict[str, int]] = {}

        for i, w in enumerate(words):
            vocab[w] = {"id": i, "tf_total": tf[w], "df": df.get(w, 0)}

        return vocab
