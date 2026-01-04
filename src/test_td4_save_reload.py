from Corpus import Corpus


# 1) Charger le TSV existant
# Objectif : vérifier que load_from_tsv marche et qu'on a des docs.
c1 = Corpus("test")
c1.load_from_tsv("data/corpus.tsv")
print("Loaded:", len(c1), "docs")

# 2) Sauver dans un nouveau fichier
# Objectif : tester save_to_tsv.
out = "data/corpus_saved.tsv"
c1.save_to_tsv(out)
print("Saved to:", out)

# 3) Recharger depuis le fichier sauvegardé
# On reset le Singleton pour être sûr de repartir d'une instance neuve.
Corpus.reset_singleton()
c2 = Corpus("test2")
c2.load_from_tsv(out)
print("Reloaded:", len(c2), "docs")

# 4) Vérifs simples
# - même nombre de docs après reload
# - ids cohérents
assert len(c1) == len(c2), "Nombre de docs différent après reload"
assert list(c2.documents.keys()) == sorted(c2.documents.keys()), "Ids non triables / incohérents"

# Vérif id 0 existe et correspond bien à un doc
assert c2.get_doc(0) is not None, "doc_id=0 manquant après reload"

print("OK: save/reload cohérent")
