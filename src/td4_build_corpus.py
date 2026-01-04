from Corpus import Corpus


if __name__ == "__main__":
    # TD4 : on veut vérifier que notre classe Corpus fonctionne bien
    # (chargement TSV, tri, auteurs, affichages, etc.)

    # 1) Initialisation du corpus
    corpus = Corpus("Corpus TD4")

    # 2) Chargement des données depuis le TSV produit au TD3
    corpus.load_from_tsv("data/corpus.tsv")

    # __repr__ du corpus (résumé pratique)
    print(corpus)

    # Test 1 : tri des documents par titre
    print("\n--- Documents triés par titre ---")
    for doc in corpus.documents_tries_par_titre(5):
        print(doc)

    # Test 2 : tri des documents par date
    print("\n--- Documents triés par date ---")
    for doc in corpus.documents_tries_par_date(5):
        print(doc)

    # Test 3 : auteurs
    # vérifier que la création des auteurs se fait automatiquement
    # vérifier que le nb de docs par auteur est cohérent
    print("\n--- Auteurs ---")
    for author in corpus.authors.values():
        print(author)

    # Test 4 : inspection détaillée d’un document
    # On prend doc_id=0 pour voir le type Python et le rendu de afficher()
    print("\n--- Inspection d’un document (doc_id=0) ---")
    doc0 = corpus.documents.get(0)
    if doc0:
        print("Type de l’objet :", type(doc0))
        print(doc0.afficher())

    # Test 5 : vérification de la structure interne
    print("\n--- Vérification structure interne ---")
    print("documents est un dict :", isinstance(corpus.documents, dict))
    print("authors est un dict :", isinstance(corpus.authors, dict))
    print("nb auteurs :", len(corpus.authors))

    # Cas particulier : auteur inconnu
    if "Unknown" in corpus.authors:
        print("Auteur 'Unknown' présent :", True)
        print("Nombre de docs pour 'Unknown' :", corpus.authors["Unknown"].nb_docs)
