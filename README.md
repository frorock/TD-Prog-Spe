# Utilisation du projet

Toutes les commandes doivent être exécutées **depuis la racine du projet**.

## 1) Installer les dépendances et créé le fichier .env
```bash
pip install -r requirements.txt
```
créer un fichier .env à la racine du projet avec : 
REDDIT_CLIENT_ID=XXX
REDDIT_CLIENT_SECRET=XXX
REDDIT_USER_AGENT=TD3


## 2) Générer ou charger le corpus (TD3)
```bash
python src/td3_acquisition.py
```
- Charge un corpus existant s’il est déjà présent (`data/corpus.tsv`)
- Sinon, collecte les données et génère le fichier TSV

## 3) Construire et tester le corpus (TD4)
```bash
python src/td4_build_corpus.py
```

Test de persistance recommandé :
```bash
python src/test_td4_save_reload.py
```

## 4) Tester la fabrique de documents (TD5)
```bash
python src/td5_build_corpus.py
```

## 5) Analyse textuelle (TD6)
```bash
python src/td6_test.py
```

## 6) Moteur de recherche TF-IDF (TD7)
```bash
python src/td7_search.py
```

## 7) Extension aux discours politiques (TD8)
```bash
python src/td8_discours.py
```

## 8) Interface utilisateur (TD9–TD10)
Ouvrir le notebook `TD9_Interface.ipynb`, exécuter les cellules puis appeler :
```python
build_widgets_ui(se)
```
