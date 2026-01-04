from Corpus import Corpus


if __name__ == "__main__":
    # 1) Création + chargement du corpus (TD5)

    corpus = Corpus("Corpus TD5")

    # Chargement des documents depuis le TSV (issu du TD3 enrichi)
    # créer des documents spécialisés via DocumentFactory
    # alimenter le dictionnaire authors via Corpus.add_document()
    corpus.load_from_tsv("data/corpus.tsv")

    # Affichage synthétique du corpus (via __repr__)
    print(corpus)

    # 2) Test : affichage des 5 premiers documents
    # corpus.documents contient des objets Document (ou sous-classes)
    # getType() renvoie "reddit"/"arxiv"
    # __str__ affiche correctement selon le type
    print("\n 5 premiers documents (type + str)")
    for i in range(min(5, len(corpus.documents))):
        # corpus.documents est un dict on accède ici par id
        doc = corpus.documents[i]
        print(doc.getType(), "=>", doc)

    # 3) Test : statistiques par type de document
    # Objectif : vérifier que la Factory a instancié la bonne classepour chaque origine (reddit/arxiv).
    print("\nStats types (reddit/arxiv)")
    nb_reddit = sum(1 for d in corpus.documents.values() if d.getType() == "reddit")
    nb_arxiv = sum(1 for d in corpus.documents.values() if d.getType() == "arxiv")
    nb_doc = sum(1 for d in corpus.documents.values() if d.getType() == "document")

    print("reddit :", nb_reddit)
    print("arxiv  :", nb_arxiv)
    print("autres :", nb_doc)

    # 4) Test : inspection détaillée d’un document Reddit
    # On cherche le premier document reddit disponible et on appelle afficher()
    print("\n Inspection d’un document reddit")
    for d in corpus.documents.values():
        if d.getType() == "reddit":
            print(d.afficher())
            break

    # 5) Test : vérification du Singleton
    print("\nTest Singleton")
    c1 = Corpus("A")
    c2 = Corpus("B")
    print("Singleton OK :", c1 is c2)
