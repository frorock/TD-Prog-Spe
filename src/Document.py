from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Document:
    """
    Classe de base (TD4) : ce qu'ont en commun tous les documents.

    Champs communs :
    - titre, auteur, date, url, texte

    Champs "extra" (TD8+ / extension) :
    - permet de stocker des infos bonus sans casser la structure
    """

    titre: str
    auteur: str
    date: Optional[datetime]
    url: str
    texte: str

    # Dictionnaire libre pour les métadonnées supplémentaires
    extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Type "logique" du document (utile pour afficher la source)
    _type: str = field(default="document", repr=False)

    def getType(self) -> str:
        """Renvoie le type de document (utilisé dans Corpus / SearchEngine)."""
        return self._type

    def afficher(self) -> str:
        """
        Affiche toutes les infos.
        Utile pour débugger ou inspecter un document.
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
        """Affichage digest (TD4) : juste titre + auteur."""
        return f"{self.titre} — {self.auteur}"


@dataclass
class RedditDocument(Document):
    """
    Classe fille (TD5) : document venant de Reddit.

    Spécificité Reddit :
    - nombre de commentaires
    """

    nb_commentaires: int = 0

    def __post_init__(self) -> None:
        # On fixe le type ici (comme demandé dans le TD5)
        self._type = "reddit"

    def __str__(self) -> str:
        return f"[Reddit] {self.titre} — {self.auteur} ({self.nb_commentaires} com)"


@dataclass
class ArxivDocument(Document):
    """
    Classe fille (TD5) : document venant d'Arxiv.

    Spécificité Arxiv :
    - liste de co-auteurs
    """

    co_auteurs: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._type = "arxiv"

    def __str__(self) -> str:
        # Petit affichage propre, sans spammer tous les noms si la liste est longue
        co = ", ".join(self.co_auteurs[:3])
        if len(self.co_auteurs) > 3:
            co += "…"
        return f"[Arxiv] {self.titre} — {self.auteur} (co: {co})"
