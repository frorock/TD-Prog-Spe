from __future__ import annotations

from typing import Any, Dict, Optional, List
import math

from Document import Document, RedditDocument, ArxivDocument


class DocumentFactory:
    """
    Factory (TD5) :
    - crée automatiquement la bonne classe de document (reddit/arxiv/generic)
    - gère les valeurs bizarres venant de pandas (ex: NaN)
    - conserve les métadonnées "extra" (utile TD8+)
    """

    def _is_nan(self, x: Any) -> bool:
        """
        Test simple pour détecter NaN (souvent présent dans des DataFrame).
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
        Crée un Document (ou une sous-classe) selon doc_type.

        doc_type attendu :
        - "reddit"
        - "arxiv"
        - sinon : Document standard
        """
        extra = extra or {}
        t = (doc_type or "document").lower().strip()

        # Cas Reddit : on récupère nb_commentaires dans extra si possible
        if t == "reddit":
            nb_raw = extra.get("nb_commentaires", 0)

            # On nettoie les valeurs invalides
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
                extra=extra,
            )

        # Cas Arxiv : on récupère la liste des co-auteurs
        if t == "arxiv":
            co_raw = extra.get("co_auteurs", [])

            # Plusieurs formats possibles :
            # - déjà une liste
            # - une string "a;b;c"
            # - NaN / None
            if co_raw is None or self._is_nan(co_raw):
                co_list: List[str] = []
            elif isinstance(co_raw, list):
                co_list = [str(x).strip() for x in co_raw if str(x).strip()]
            else:
                s = str(co_raw).strip()
                co_list = [x.strip() for x in s.split(";") if x.strip()] if s else []

            return ArxivDocument(
                titre=titre,
                auteur=auteur,
                date=date,
                url=url,
                texte=texte,
                co_auteurs=co_list,
                extra=extra,
            )

        # Sinon : document générique
        return Document(
            titre=titre,
            auteur=auteur,
            date=date,
            url=url,
            texte=texte,
            extra=extra,
        )
