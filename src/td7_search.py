from Corpus import Corpus
from SearchEngine import SearchEngine


if __name__ == "__main__":
    # Chargement du corpus (issu TD3 enrichi + TD5)
    corpus = Corpus("Corpus TD7")
    corpus.load_from_tsv("data/corpus.tsv")

    print(corpus)

    # Construction du moteur (TD7)
    engine = SearchEngine(corpus)

    print("\n--- TD7 : aperçu vocab ---")
    vocab_df = engine.get_vocab_df()
    print("Taille vocab :", len(vocab_df))
    print(vocab_df.head(10))

    print("\n--- TD7 : dimensions matrices ---")
    print("matTF shape     :", engine.matTF.shape)
    print("matTFxIDF shape :", engine.matTFxIDF.shape)

    # Test recherche
    query = "climate change carbon emissions"
    print("\n--- TD7 : recherche ---")
    print("Query :", query)
    res = engine.search(query, top_k=10)
    print(res)
