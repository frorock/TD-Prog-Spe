from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Author:
    """
    Représente un auteur dans le corpus.

    Idée (TD4) :
    - un auteur a un nom
    - on veut compter combien de documents il a produits
    - on garde aussi la liste de ses documents (via leur id)
    """

    name: str
    nb_docs: int = 0

    # production : dictionnaire {titre_du_doc: [liste_des_doc_id]}
    # pratique pour retrouver rapidement quels docs appartiennent à l'auteur
    production: Dict[str, List[int]] = field(default_factory=dict)

    def add(self, doc_id: int, titre: str) -> None:
        """
        Ajoute un document (identifié par doc_id) dans la production de l'auteur.

        Appel typique :
        - cette méthode est appelée depuis Corpus.add_document()
        """
        self.nb_docs += 1

        # Si le titre n'existe pas encore, on crée la liste
        if titre not in self.production:
            self.production[titre] = []

        # On ajoute l'id du document dans la liste associée au titre
        self.production[titre].append(doc_id)

    def __str__(self) -> str:
        """Affichage simple quand on fait print(author)."""
        return f"{self.name} ({self.nb_docs} documents)"
