from pathlib import Path
from Corpus import Corpus

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    tsv_path = project_root / "data" / "corpus.tsv"

    corpus = Corpus("Corpus TD6")
    corpus.load_from_tsv(str(tsv_path))

    print(corpus)

    print("\nTD6 stats")
    freq = corpus.stats(n=10)
    # au cas où stats() retourne quelque chose
    if freq is not None:
        print(freq)

    print("\nTD6 concorde")
    res = corpus.concorde("climate", taille_contexte=30, n=5)

    # Si concorde() ne fait pas de print, on affiche ici
    if res is None:
        print("concorde() a renvoyé None.")
    else:
        print(res)
