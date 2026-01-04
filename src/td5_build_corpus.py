from Corpus import Corpus


if __name__ == "__main__":
    # TD5 : on ajoute l'idée de "documents spécialisés" (reddit/arxiv)
    # via une Factory qui instancie RedditDocument / ArxivDocument.

    # 1) Création + chargement du corpus
    corpus = Corpus("Corpus TD5")

    # Le TSV vient du TD3 (avec colonnes nb_commentaires et co_auteurs)
    corpus.load_from_tsv("data/corpus.tsv")

    # Résumé du corpus
    print(corpus)

    # 2) Test : affichage des 5 premiers documents
    # Ici on vérifie :
    # - getType() renvoie bien reddit/arxiv
    # - __str__ affiche comme prévu selon la classe
    print("\n--- 5 premiers documents (type + str) ---")
    for i in range(min(5, len(corpus.documents))):
        doc = corpus.documents[i]  # corpus.documents est un dict, accès par id
        print(doc.getType(), "=>", doc)

    # 3) Test : statistiques par type de document
    # Objectif : vérifier que la Factory a bien créé les bonnes classes.
    print("\n--- Stats types (reddit/arxiv) ---")
    nb_reddit = sum(1 for d in corpus.documents.values() if d.getType() == "reddit")
    nb_arxiv = sum(1 for d in corpus.documents.values() if d.getType() == "arxiv")
    nb_doc = sum(1 for d in corpus.documents.values() if d.getType() == "document")

    print("reddit :", nb_reddit)
    print("arxiv  :", nb_arxiv)
    print("autres :", nb_doc)

    # 4) Test : inspection détaillée d’un document Reddit
    # On cherche le premier reddit dispo et on affiche en format complet.
    print("\n--- Inspection d’un document reddit (si dispo) ---")
    for d in corpus.documents.values():
        if d.getType() == "reddit":
            print(d.afficher())
            break

    # 5) Test : vérification du Singleton
    print("\n--- Test Singleton ---")
    c1 = Corpus("A")
    c2 = Corpus("B")
    print("Singleton OK :", c1 is c2)
