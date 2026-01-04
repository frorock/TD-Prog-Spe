from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Author:
    """
    Représentation d'un auteur dans le corpus (TD4 / TD5)

    Cette classe permet de :
      - représenter un auteur identifié dans le corpus
      - compter le nombre de documents qu'il a produits
      - conserver la liste de ses productions (documents associés)
    """

    name: str
    nb_docs: int = 0
    production: Dict[str, List[int]] = field(default_factory=dict)

    def add(self, doc_id: int, titre: str) -> None:
        """
        Associe un document à l'auteur.

        Cette méthode est appelée automatiquement par Corpus.add_document().
        """
        self.nb_docs += 1

        if titre not in self.production:
            self.production[titre] = []

        self.production[titre].append(doc_id)

    def __str__(self) -> str:
        """
        Représentation lisible d'un auteur
        """
        return f"{self.name} ({self.nb_docs} documents)"
