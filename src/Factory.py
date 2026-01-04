from __future__ import annotations

from typing import Any, Dict, Optional, List
import math

from Document import Document, RedditDocument, ArxivDocument


class DocumentFactory:
    """
    Cette Factory centralise la logique d'instanciation des documentsafin de :
      - découpler la création des objets de leur utilisation
      - instancier dynamiquement la bonne sous-classe de Document
      - isoler la gestion des cas particuliers liés aux données (NaN, types hétérogènes)

    Dans le cadre du TD5, cette classe illustre :
      - le pattern de conception Factory
      - le polymorphisme (retour d'un type Document, mais instance concrète spécialisée)
      - une gestion robuste de données issues de sources externes (Reddit / Arxiv )
    """

    def _is_nan(self, x: Any) -> bool:
        """
        Détecte si une valeur correspond à un NaN
        """
        try:
            return isinstance(x, float) and math.isnan(x)
        except Exception:
            return False

    def create(
        self,
        doc_type: str,
        titre: str,
        auteur: str,
        date,
        url: str,
        texte: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Méthode principale de la Factory : crée et retourne un Document approprié.

        """
        # Si extra est None on travaille avec un dict vide
        extra = extra or {}

        # Normalisation du type pour éviter les erreurs dues à la casse ou aux espaces
        t = (doc_type or "document").lower().strip()

        # Cas 1 : Document Reddit
        if t == "reddit":
            nb_raw = extra.get("nb_commentaires", 0)

            if nb_raw is None or self._is_nan(nb_raw):
                nb = 0
            else:
                try:
                    nb = int(nb_raw)
                except Exception:
                    nb = 0

            return RedditDocument(
                titre=titre,
                auteur=auteur,
                date=date,
                url=url,
                texte=texte,
                nb_commentaires=nb,
            )

        # Cas 2 : Document Arxiv
        if t == "arxiv":
            co_raw = extra.get("co_auteurs", [])

            if co_raw is None or self._is_nan(co_raw):
                co_list: List[str] = []
            elif isinstance(co_raw, list):
                # Nettoyage de la liste existante
                co_list = [str(x).strip() for x in co_raw if str(x).strip()]
            else:
                # Cas le plus courant : chaîne de type "A;B;C"
                s = str(co_raw).strip()
                if not s:
                    co_list = []
                else:
                    co_list = [x.strip() for x in s.split(";") if x.strip()]

            return ArxivDocument(
                titre=titre,
                auteur=auteur,
                date=date,
                url=url,
                texte=texte,
                co_auteurs=co_list,
            )

        # Cas 3 : Document générique (fallback)
        return Document(
            titre=titre,
            auteur=auteur,
            date=date,
            url=url,
            texte=texte,
        )
