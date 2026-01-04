from pathlib import Path
from Corpus import Corpus


if __name__ == "__main__":
    # TD6 : on teste surtout les méthodes "stats" et "concorde"

    # Chemin robuste vers data/corpus.tsv
    project_root = Path(__file__).resolve().parent.parent
    tsv_path = project_root / "data" / "corpus.tsv"

    # Chargement corpus
    corpus = Corpus("Corpus TD6")
    corpus.load_from_tsv(str(tsv_path))

    print(corpus)

    # 1) Test stats vocab
    print("\n--- TD6 stats ---")
    freq = corpus.stats(n=10)

    # Selon comment tu as codé stats(), ça peut print ou retourner un DataFrame
    if freq is not None:
        print(freq)

    # 2) Test concorde
    print("\n--- TD6 concorde ---")
    res = corpus.concorde("climate", taille_contexte=30, n=5)

    # Pareil : si concorde() print directement, res peut être None
    if res is None:
        print("concorde() a renvoyé None.")
    else:
        print(res)
