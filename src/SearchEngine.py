from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import math

import pandas as pd
from scipy.sparse import csr_matrix

from Corpus import Corpus


@dataclass
class SearchEngine:
    """
    TD7 – Moteur de recherche vectoriel (TF / TF-IDF)

    Objectifs (conformes énoncé) :
    - Construire un vocabulaire `vocab` : dict[mot] -> infos (id, tf_total, df, idf)
    - Construire la matrice `matTF` (docs x termes) en sparse (CSR)
    - Construire la matrice `matTFxIDF` (docs x termes) en sparse (CSR)
    - Permettre la recherche : transformation requête -> vecteur -> similarité cosinus
    - Retourner les résultats dans un DataFrame pandas
    """

    corpus: Corpus
    vocab: Dict[str, Dict[str, Any]]
    matTF: csr_matrix
    matTFxIDF: csr_matrix
    doc_norms_tfidf: List[float]


    idf_by_id: List[float]

    # Construction
    def __init__(self, corpus: Corpus) -> None:
        """
        À l’instanciation :
        - construit vocab
        - construit matTF
        - calcule df, tf_total
        - construit matTFxIDF
        - pré-calcule les normes des documents (pour cosinus)
        """
        self.corpus = corpus
        self.vocab = {}
        self.matTF = csr_matrix((0, 0), dtype=float)
        self.matTFxIDF = csr_matrix((0, 0), dtype=float)
        self.doc_norms_tfidf = []
        self.idf_by_id = []

        self._build()

    def _build(self) -> None:
        """
        Pipeline TD7 complet :
        1) tokenisation de tous les docs
        2) construction vocab (ids)
        3) construction matTF (CSR)
        4) calcul df / tf_total
        5) calcul idf
        6) construction matTFxIDF
        7) pré-calcul des normes docs (cosinus)
        """
        docs_tokens: List[List[str]] = []
        for doc in self.corpus.iter_docs():
            docs_tokens.append(self.corpus.tokenize(doc.texte))

        n_docs = len(docs_tokens)

        # Sécurité : corpus vide => moteur vide
        if n_docs == 0:
            self.vocab = {}
            self.matTF = csr_matrix((0, 0), dtype=float)
            self.matTFxIDF = csr_matrix((0, 0), dtype=float)
            self.doc_norms_tfidf = []
            self.idf_by_id = []
            return

        # 1) vocab : assigner un id à chaque mot (tri alphabétique pour stabilité)
        all_words = sorted({w for tokens in docs_tokens for w in tokens})
        self.vocab = {
            w: {"id": i, "tf_total": 0, "df": 0, "idf": 0.0}
            for i, w in enumerate(all_words)
        }
        n_terms = len(self.vocab)

        # Sécurité : aucun terme (cas extrême)
        if n_terms == 0:
            self.matTF = csr_matrix((n_docs, 0), dtype=float)
            self.matTFxIDF = csr_matrix((n_docs, 0), dtype=float)
            self.doc_norms_tfidf = [0.0] * n_docs
            self.idf_by_id = []
            return

        # 2) construire matTF en CSR via (data, row, col)
        data: List[float] = []
        rows: List[int] = []
        cols: List[int] = []

        # stats pour df / tf_total
        tf_total = [0] * n_terms
        df = [0] * n_terms

        for doc_id, tokens in enumerate(docs_tokens):
            if not tokens:
                continue

            # TF doc : compter occurrences dans le doc
            local_tf: Dict[int, int] = {}
            for w in tokens:
                term_id = self.vocab[w]["id"]
                local_tf[term_id] = local_tf.get(term_id, 0) + 1

            # remplir matrice
            for term_id, tf in local_tf.items():
                rows.append(doc_id)
                cols.append(term_id)
                data.append(float(tf))

                tf_total[term_id] += tf
                df[term_id] += 1

        self.matTF = csr_matrix((data, (rows, cols)), shape=(n_docs, n_terms), dtype=float)

        # 3) stocker df/tf_total dans vocab
        for w, info in self.vocab.items():
            tid = info["id"]
            info["tf_total"] = int(tf_total[tid])
            info["df"] = int(df[tid])

        # 4) calcul idf (log(N/df))
        # (si df==0 -> idf=0 par sécurité)
        self.idf_by_id = [0.0] * n_terms
        for w, info in self.vocab.items():
            dfi = info["df"]
            if dfi <= 0:
                idf_val = 0.0
            else:
                idf_val = math.log(n_docs / dfi)

            info["idf"] = float(idf_val)
            self.idf_by_id[info["id"]] = float(idf_val)

        # 5) matTFxIDF = matTF * idf(term)
        # On multiplie chaque colonne par son idf
        diag_rows = list(range(n_terms))
        diag_cols = list(range(n_terms))
        diag_data = self.idf_by_id
        diag = csr_matrix((diag_data, (diag_rows, diag_cols)), shape=(n_terms, n_terms), dtype=float)

        self.matTFxIDF = self.matTF.dot(diag)

        # 6) normes documents (cosinus) sur TF-IDF
        # norme = sqrt(sum(x_i^2))
        squared = self.matTFxIDF.multiply(self.matTFxIDF).sum(axis=1)
        self.doc_norms_tfidf = [math.sqrt(float(s)) for s in squared.A1]

    # Outils TD7
    def get_vocab_df(self) -> pd.DataFrame:
        """
        Retourne le vocabulaire sous forme DataFrame trié par id.
        Colonnes : mot, id, tf_total, df, idf
        """
        rows = []
        for mot, info in self.vocab.items():
            rows.append(
                {
                    "mot": mot,
                    "id": info["id"],
                    "tf_total": info["tf_total"],
                    "df": info["df"],
                    "idf": info["idf"],
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values(by="id", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def _query_to_vec(self, query: str) -> Tuple[csr_matrix, float]:
        """
        Transforme une requête en vecteur TF-IDF (1 x n_terms) compatible vocab.
        Retourne (vec, norme_vec).
        """
        tokens = self.corpus.tokenize(query)
        if not tokens or len(self.vocab) == 0:
            return csr_matrix((1, len(self.vocab)), dtype=float), 0.0

        local_tf: Dict[int, int] = {}
        for w in tokens:
            if w not in self.vocab:
                continue
            tid = self.vocab[w]["id"]
            local_tf[tid] = local_tf.get(tid, 0) + 1

        if not local_tf:
            return csr_matrix((1, len(self.vocab)), dtype=float), 0.0

        rows = []
        cols = []
        data = []
        for tid, tf in local_tf.items():
            # TF-IDF(query) = tf * idf
            idf = float(self.idf_by_id[tid]) if tid < len(self.idf_by_id) else 0.0
            rows.append(0)
            cols.append(tid)
            data.append(float(tf) * idf)

        vec = csr_matrix((data, (rows, cols)), shape=(1, len(self.vocab)), dtype=float)
        norm = math.sqrt(float(vec.multiply(vec).sum()))
        return vec, norm

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Recherche vectorielle TF-IDF + cosinus.

        Retourne un DataFrame avec :
        - rank
        - score
        - doc_id
        - origine
        - titre
        - auteur
        - date
        - url
        """
        if top_k <= 0:
            return pd.DataFrame(columns=["rank", "score", "doc_id", "origine", "titre", "auteur", "date", "url"])

        qvec, qnorm = self._query_to_vec(query)
        if qnorm == 0.0 or len(self.corpus.documents) == 0:
            return pd.DataFrame(
                columns=["rank", "score", "doc_id", "origine", "titre", "auteur", "date", "url"]
            )

        # produit scalaire : (n_docs x n_terms) dot (n_terms x 1) => (n_docs x 1)
        scores = self.matTFxIDF.dot(qvec.T)  # sparse
        scores = scores.toarray().ravel().tolist()

        # cosinus : score / (||doc|| * ||query||)
        results: List[Tuple[int, float]] = []
        for doc_id, dot in enumerate(scores):
            dnorm = self.doc_norms_tfidf[doc_id] if doc_id < len(self.doc_norms_tfidf) else 0.0
            if dnorm == 0.0:
                continue
            cos = float(dot) / (dnorm * qnorm)
            if cos > 0:
                results.append((doc_id, cos))

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        # construire DataFrame final en récupérant métadonnées du doc
        rows = []
        for rank, (doc_id, score) in enumerate(results, start=1):
            doc = self.corpus.documents.get(doc_id)
            if doc is None:
                continue
            rows.append(
                {
                    "rank": rank,
                    "score": score,
                    "doc_id": doc_id,
                    "origine": doc.getType(),
                    "titre": doc.titre,
                    "auteur": doc.auteur,
                    "date": doc.date.isoformat() if doc.date else "",
                    "url": doc.url,
                }
            )

        return pd.DataFrame(rows)
