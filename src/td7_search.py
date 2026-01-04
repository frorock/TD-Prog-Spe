from Corpus import Corpus
from SearchEngine import SearchEngine


if __name__ == "__main__":
    # TD7 : on construit le moteur de recherche TF / TF-IDF puis on teste une requête.

    # 1) Chargement du corpus (TSV issu du TD3 + enrichi au TD5)
    corpus = Corpus("Corpus TD7")
    corpus.load_from_tsv("data/corpus.tsv")

    print(corpus)

    # 2) Construction du moteur (build vocab + matrices)
    engine = SearchEngine(corpus)

    # 3) Aperçu vocabulaire
    print("\n--- TD7 : aperçu vocab ---")
    vocab_df = engine.get_vocab_df()
    print("Taille vocab :", len(vocab_df))
    print(vocab_df.head(10))

    # 4) Dimensions des matrices
    print("\n--- TD7 : dimensions matrices ---")
    print("matTF shape     :", engine.matTF.shape)
    print("matTFxIDF shape :", engine.matTFxIDF.shape)

    # 5) Test recherche
    query = "climate change carbon emissions"
    print("\n--- TD7 : recherche ---")
    print("Query :", query)
    res = engine.search(query, top_k=10)
    print(res)
