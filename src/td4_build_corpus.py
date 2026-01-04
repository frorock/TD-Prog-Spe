from Corpus import Corpus


if __name__ == "__main__":
    # Initialisation du corpus
    corpus = Corpus("Corpus TD4")

    # Chargement des données depuis le TSV (TD3)
    corpus.load_from_tsv("data/corpus.tsv")

    # Affichage du résumé du corpus (via __repr__)
    print(corpus)

    # Test 1 : tri des documents par titre
    # Vérifie que les documents sont bien stockés et que la méthode de tri fonctionne
    print("\n--- Documents triés par titre ---")
    for doc in corpus.documents_tries_par_titre(5):
        print(doc)

    # Test 2 : tri des documents par date
    print("\n--- Documents triés par date ---")
    for doc in corpus.documents_tries_par_date(5):
        print(doc)

    # Test 3 : auteurs
    # Vérifie la création automatique des auteurs etle comptage du nombre de documents par auteur
    print("\n--- Auteurs ---")
    for author in corpus.authors.values():
        print(author)

    # Test 4 : inspection détaillée d’un document
    # On inspecte explicitement le document d’id 0 afin de vérifier son type Python et le bon fonctionnement de la méthode afficher()
    print("\n--- Inspection d’un document (doc_id=0) ---")
    doc0 = corpus.documents.get(0)
    if doc0:
        print("Type de l’objet :", type(doc0))
        print(doc0.afficher())

    # Test 5 : vérification de la structure interne
    # vérifie que documents et auteurs sont stockés dans un dictionnaire
    print("\n--- Vérification structure interne ---")
    print("documents est un dict :", isinstance(corpus.documents, dict))
    print("authors est un dict :", isinstance(corpus.authors, dict))
    print("nb auteurs :", len(corpus.authors))

    # Cas particulier : auteur inconnu
    if "Unknown" in corpus.authors:
        print("Auteur 'Unknown' présent :", True)
        print("Nombre de docs pour 'Unknown' :", corpus.authors["Unknown"].nb_docs)
