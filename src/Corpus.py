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
    Convertit une date (string) en objet datetime de manière robuste.

    Objectif :
    - éviter que le chargement du TSV casse si un document n'a pas de date
    - gérer les formats typiques rencontrés dans nos sources :
        * Reddit : ISO8601 avec timezone (ex: 2025-07-27T09:53:49+00:00)
        * Arxiv  : ISO-like avec parfois un suffixe 'Z' (UTC)

    Retour :
    - datetime si conversion possible
    - None sinon (format invalide, vide, etc.)
    """
    # Cas None ou chaîne vide => pas de date
    if not date_str:
        return None

    # Suppression des espaces
    s = str(date_str).strip()
    if not s:
        return None

    # Arxiv peut fournir un suffixe "Z" (UTC) et fromisoformat() ne le supporte pas
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    # On retourne None si format invalide
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


@dataclass
class Corpus:
    """
    Rôle de la classe :
    - Stocker un ensemble de documents (instances de Document ou de ses sous-classes)
    - Maintenir une structure "auteurs" (dictionnaire d'Author)
    - Fournir des méthodes de tri / inspection
    - Implémenter un Singleton

    TD6 / TD7 :
    - Ajouter des outils de traitement de texte (nettoyage, stats, vocabulaire)
    - Fournir une tokenisation stable et cohérente (utile pour TF / TF-IDF)
    """

    nom: str
    documents: Dict[int, Document] = field(default_factory=dict)
    authors: Dict[str, Author] = field(default_factory=dict)
    _next_id: int = 0

    # Cache de la chaîne globale du corpus.
    # Permet d'éviter de reconstruire la concaténation
    # de tous les textes à chaque appel (optimisation demandée).
    _global_str: Optional[str] = field(default=None, init=False, repr=False)

    #  Singleton
    _instance: Optional["Corpus"] = None

    def __new__(cls, *args, **kwargs):
        """
        Implémentation du pattern Singleton.
        - On ne veut qu'une seule instance du Corpus dans l'application.
        - Si on rappelle Corpus("X") plus tard, on récupère la même instance.
        """
        if cls._instance is None:
            cls._instance = super(Corpus, cls).__new__(cls)
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Réinitialise le singleton.
        Utile en tests si on veut repartir d'un Corpus vide.
        """
        cls._instance = None

    # Méthodes métier
    def add_document(self, doc: Document) -> int:
        """
        Ajoute un document au corpus et met à jour la structure auteurs.

        Effets :
        - stocke le document dans self.documents avec un id interne
        - incrémente _next_id
        - met à jour self.authors : création de l'auteur si besoin,
          et ajout d'une référence vers ce document.
        """
        doc_id = self._next_id
        self.documents[doc_id] = doc
        self._next_id += 1

        # Gestion d'un auteur par défaut si la source ne le fournit pas
        author_name = doc.auteur or "Unknown"
        if author_name not in self.authors:
            self.authors[author_name] = Author(author_name)

        # L'auteur référence les documents via doc_id, titre
        self.authors[author_name].add(doc_id, doc.titre)

        # Le corpus a changé => on invalide le cache global_str
        self._global_str = None

        return doc_id

    # AJOUTS TD6/TD7 (V2)
    def iter_docs(self) -> Iterator[Document]:
        """
        Itère sur les documents du corpus.

        - SearchEngine attend une méthode iter_docs()
        - on renvoie les documents dans l'ordre des doc_id (0..N-1)
        """
        for doc_id in sorted(self.documents.keys()):
            yield self.documents[doc_id]

    def iter_docs_items(self) -> Iterator[Tuple[int, Document]]:
        """
        Variante utile : itère sur (doc_id, doc).
        """
        for doc_id in sorted(self.documents.keys()):
            yield doc_id, self.documents[doc_id]

    def get_doc(self, doc_id: int) -> Optional[Document]:
        """
        Accès sûr à un document.
        Permet d'éviter des KeyError dans les scripts/tests.
        """
        return self.documents.get(doc_id)

    def add_document_with_id(self, doc_id: int, doc: Document) -> None:
        """
        Ajoute un document avec un id imposé (utile quand on recharge depuis un TSV).

        Effets :
        - stocke le document à l'id donné
        - met à jour _next_id pour éviter collisions
        - met à jour authors
        - invalide le cache global_str

        Remarque :
        - on garde add_document() pour le mode "création" (API),
          et on utilise add_document_with_id() pour le mode "chargement TSV".
        """
        doc_id = int(doc_id)
        self.documents[doc_id] = doc
        self._next_id = max(self._next_id, doc_id + 1)

        author_name = doc.auteur or "Unknown"
        if author_name not in self.authors:
            self.authors[author_name] = Author(author_name)

        self.authors[author_name].add(doc_id, doc.titre)

        self._global_str = None


    def documents_tries_par_titre(self, n: int = 10) -> List[Document]:
        """
        Renvoie les n premiers documents triés par titre (ordre alphabétique).
        """
        return sorted(self.documents.values(), key=lambda d: (d.titre or "").lower())[:n]

    def documents_tries_par_date(self, n: int = 10) -> List[Document]:
        """
        Renvoie les n documents les plus récents selon la date.
        (si date absente : datetime.min)
        """
        return sorted(
            self.documents.values(),
            key=lambda d: d.date if d.date else datetime.min,
            reverse=True,
        )[:n]

    def __repr__(self) -> str:
        """
        Représentation concise du corpus (pratique pour debug/tests).
        """
        return f"Corpus(nom={self.nom}, nb_documents={len(self.documents)}, nb_auteurs={len(self.authors)})"

    # TD4 : Sauvegarde TSV 
    def save_to_tsv(self, path: str) -> None:
        """
        TD4 - Sauvegarde du corpus au format TSV.

        Objectif :
        - pouvoir recharger le corpus sans refaire les appels API
        - garder une structure cohérente (doc_id stable)

        """
        rows: List[Dict[str, Any]] = []

        for doc_id, doc in self.iter_docs_items():
            origine = ""
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

            # - row.to_dict() (au chargement) contient déjà les colonnes de base
            # donc ici, on n'exporte que les vraies colonnes additionnelles,
            #   et on ignore explicitement les colonnes "de base".
            base_cols = set(row.keys())
            extra = getattr(doc, "extra", None)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if k in base_cols:
                        continue
                    # petit nettoyage : certaines libs peuvent mettre des NaN
                    if isinstance(v, float) and pd.isna(v):
                        v = ""
                    row[k] = v

            rows.append(row)

        df = pd.DataFrame(rows)

        # Petite finition : colonnes de base en premier
        base_cols_order = ["id", "origine", "titre", "auteur", "date", "url", "texte"]
        other_cols = [c for c in df.columns if c not in base_cols_order]
        df = df[base_cols_order + other_cols]

        df.to_csv(path, sep="\t", index=False)

    # Chargement depuis TSV (TD3 enrichi)
    def load_from_tsv(self, path: str) -> None:
        """
        Charge le TSV produit au TD3 enrichi.

        On crée des RedditDocument / ArxivDocument via la Factory,
        en utilisant la colonne 'origine' et les métadonnées :
        - titre
        - auteur
        - date
        - url
        - texte
        """
        df = pd.read_csv(path, sep="\t")

        # Reset du contenu
        self.documents = {}
        self.authors = {}
        self._next_id = 0
        self._global_str = None

        factory = DocumentFactory()

        for _, row in df.iterrows():
            # `origine` pilote la Factory (reddit/arxiv)
            origine = str(row.get("origine", "document")).lower().strip()

            # Titre
            titre = str(row.get("titre", "") or "").strip()
            if not titre:
                titre = f"Document {int(row.get('id', self._next_id))}"

            # Auteur
            auteur = str(row.get("auteur", "") or "").strip() or "Unknown"

            # Date
            date_raw = str(row.get("date", "") or "").strip()
            date = parse_date_safe(date_raw)

            # URL / texte
            url = str(row.get("url", "") or "").strip()
            texte = str(row.get("texte", "") or "").strip()

            # Création du document via Factory
            doc = factory.create(
                doc_type=origine,
                titre=titre,
                auteur=auteur,
                date=date,
                url=url,
                texte=texte,
                extra=row.to_dict(),
            )

            # - on conserve l'id provenant du TSV
            # - ça garantit que doc_id == id interne (cohérence matrices/recherche)
            doc_id = int(row.get("id", self._next_id))

            # Ajout au corpus + indexation par auteur
            self.add_document_with_id(doc_id, doc)

    # TD6 : Nettoyage + stats + concordancier + recherche regex
    def nettoyer_texte(self, s: str) -> str:
        """
        Nettoyage minimal (TD6) :
        - minuscules
        - remplacement retours à la ligne
        - suppression chiffres
        - suppression ponctuation (remplacement par espace)
        - normalisation des espaces
        """
        if s is None:
            return ""

        s = str(s).lower()
        s = s.replace("\n", " ").replace("\r", " ")
        s = re.sub(r"\d+", " ", s)

        # la ponctuation doit servir de délimiteur on supprime donc tout caractère non alphabétique,
        s = re.sub(r"[^a-z\s]+", " ", s)

        s = re.sub(r"\s+", " ", s).strip()
        return s

    def iter_docs_text(self) -> List[str]:
        """
        Retourne la liste des textes des documents.
        Méthode utilitaire utilisée dans plusieurs TD (stats, vocab, TF/TF-IDF).
        """
        return [d.texte for d in self.iter_docs()]

    def __len__(self) -> int:
        """
        - permet d'écrire len(corpus) pour obtenir le nombre de documents.
        - pratique en tests et rend l'API plus "Pythonique".
        """
        return len(self.documents)

    def build_global_string(self) -> str:
        """
        Construit une chaîne globale = concaténation de tous les textes du corpus.

        Optimisation (TD6) :
        - on utilise un cache (_global_str)
        - si le corpus n'a pas changé, on ne recalcule pas.
        """
        if self._global_str is None:
            # - on concatène brut (pour conserver le texte original)
            # - et on garde l'ordre stable via iter_docs_text()
            self._global_str = " ".join(self.iter_docs_text())
        return self._global_str

    def stats(self, n: int = 10) -> pd.DataFrame:
        """
        TD6 - statistiques de vocabulaire.

        Affiche/retourne :
        - nombre de mots différents
        - les n mots les plus fréquents

        On retourne un DataFrame pandas (freq) pour coller à la consigne.
        Colonnes typiques :
        - mot
        - tf_total (nombre total d'occurrences dans le corpus)
        - df (nombre de documents contenant le mot)
        """
        docs = self.iter_docs_text()

        # On construit vocab + comptages en une passe
        tf: Dict[str, int] = {}
        df: Dict[str, int] = {}

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

        # Affichages demandés (console)
        print("Nombre de mots différents dans le corpus :", len(tf))
        print(f"Top {n} mots les plus fréquents :")
        print(freq.head(n))

        return freq

    def search(self, motif: str, taille_contexte: int = 30, n: int = 10) -> pd.DataFrame:
        """
        Principe :
        - on construit une chaîne globale concaténée (cache)
        - on cherche avec `re.finditer`
        - on renvoie un DataFrame (pandas) des occurrences + contexte
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
        Principe :
        - recherche des occurrences du mot (en tant que "mot" => bornes \\b)
        - utilisation de re (obligatoire dans l'énoncé)
        - renvoie un DataFrame (pandas) avec :
            * contexte_gauche
            * motif_trouve
            * contexte_droit

        """
        g = self.build_global_string()
        target = (mot or "").strip()
        if not target:
            return pd.DataFrame(columns=["contexte_gauche", "motif_trouve", "contexte_droit"])

        # On force la recherche en "mot entier"
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

    # TD7 : Tokenisation + vocab (utilisée aussi dans SearchEngine)
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenisation simple (TD7) :
        - lower
        - suppression chiffres
        - suppression ponctuation (apostrophe incluse)
        - split par espaces
        """
        if not text:
            return []
        s = text.lower()
        s = re.sub(r"\d+", " ", s)
        s = re.sub(r"[^a-z\s]+", " ", s)  # on supprime aussi l'apostrophe
        s = re.sub(r"\s+", " ", s).strip()
        return s.split()

    def build_vocab(self) -> Dict[str, Dict[str, int]]:
        """
        TD7 : construit le vocabulaire (dictionnaire) à partir des documents.
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

        words = sorted(tf.keys())  # ordre alphabétique demandé
        vocab: Dict[str, Dict[str, int]] = {}

        for i, w in enumerate(words):
            vocab[w] = {"id": i, "tf_total": tf[w], "df": df.get(w, 0)}

        return vocab
