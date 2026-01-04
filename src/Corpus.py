from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd

from Document import Document
from Author import Author
from Factory import DocumentFactory


def parse_date_safe(date_str: str) -> Optional[datetime]:
    """
    Convertit une date (string) en objet datetime de manière robuste.
    """
    # Cas None ou chaîne vide => pas de date
    if not date_str:
        return None

    # Suppression des espaces
    s = str(date_str).strip()
    if not s:
        return None

    # Arxiv peut fournir un suffixe "Z" et fromisoformat() ne le supporte pas
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
    """

    nom: str
    documents: Dict[int, Document] = field(default_factory=dict)
    authors: Dict[str, Author] = field(default_factory=dict)
    _next_id: int = 0

    # Singleton
    _instance: Optional["Corpus"] = None

    def __new__(cls, *args, **kwargs):
        """
        Implémentation du pattern Singleton.
        """
        if cls._instance is None:
            cls._instance = super(Corpus, cls).__new__(cls)
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Réinitialise le singleton
        """
        cls._instance = None

    # Méthodes métier
    def add_document(self, doc: Document) -> int:
        """
        Ajoute un document au corpus et met à jour la structure auteurs.
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

        return doc_id

    def documents_tries_par_titre(self, n: int = 10) -> List[Document]:
        """
        Renvoie les n premiers documents triés par titre
        """
        return sorted(self.documents.values(), key=lambda d: (d.titre or "").lower())[:n]

    def documents_tries_par_date(self, n: int = 10) -> List[Document]:
        """
        Renvoie les n documents les plus récents selon la date.
        """
        return sorted(
            self.documents.values(),
            key=lambda d: d.date if d.date else datetime.min,
            reverse=True,
        )[:n]

    def __repr__(self) -> str:
        """
        Représentation concise du corpus
        """
        return (
            f"Corpus(nom=Corpus TD5, nb_documents={len(self.documents)}, nb_auteurs={len(self.authors)})"
        )

    # Chargement depuis TSV
    def load_from_tsv(self, path: str) -> None:
        """
        Charge le TSV produit au TD3 enrichi 
        """
        df = pd.read_csv(path, sep="\t")

        # Reset du contenu
        self.documents = {}
        self.authors = {}
        self._next_id = 0

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

            # Ajout au corpus + indexation par auteur
            self.add_document(doc)
