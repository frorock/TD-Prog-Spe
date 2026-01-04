from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class Document:
    """
    Cette classe constitue la racine de la hiérarchie de documents, elle modélise les attributs communs à toutes les sources :
      - titre
      - auteur
      - date
      - url
      - contenu textuel
    """

    titre: str
    auteur: str
    date: Optional[datetime]
    url: str
    texte: str
    _type: str = field(default="document", repr=False)

    def getType(self) -> str:
        """
        Retourne le type logique du document.
        """
        return self._type

    def afficher(self) -> str:
        """
        Affichage détaillé et formaté du document.
        """
        date_str = self.date.isoformat() if self.date else "N/A"

        return (
            f"[{self.getType().upper()}]\n"
            f"Titre : {self.titre}\n"
            f"Auteur : {self.auteur}\n"
            f"Date : {date_str}\n"
            f"URL : {self.url}\n"
            f"Texte : {self.texte}\n"
        )

    def __str__(self) -> str:
        """
        Représentation courte et lisible d'un document.
        """
        return f"{self.titre} — {self.auteur}"


@dataclass
class RedditDocument(Document):
    """
    Cette classe illustre :
    - l'héritage
    - l'ajout de champs spécifiques à une source
    - le polymorphisme (redéfinition de __str__)
    """

    nb_commentaires: int = 0

    def __post_init__(self) -> None:
        """
        Méthode appelée automatiquement après l'initialisation du dataclass qui permet ici de définir le type interne du document
        sans modifier le constructeur hérité.
        """
        self._type = "reddit"

    def __str__(self) -> str:
        """
        Représentation courte spécifique à Reddit
        """
        return f"[Reddit] {self.titre} — {self.auteur} ({self.nb_commentaires} com)"


@dataclass
class ArxivDocument(Document):
    """
    Cette classe modélise un article scientifique issu d'Arxiv, avec la notion de co-auteurs en plus de l'auteur principal.
    """

    co_auteurs: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Initialisation du type interne après construction.
        """
        self._type = "arxiv"

    def __str__(self) -> str:
        """
        Représentation courte spécifique à Arxiv.
        """
        co = ", ".join(self.co_auteurs[:3])
        if len(self.co_auteurs) > 3:
            co += "…"

        return f"[Arxiv] {self.titre} — {self.auteur} (co: {co})"
