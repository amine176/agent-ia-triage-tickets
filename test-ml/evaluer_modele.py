"""
evaluer_modele.py
-----------------
Illustre 2 concepts de ML manquants dans classifier.py :
  1. Train/test split : on n'entraine JAMAIS sur 100% des donnees si on veut
     evaluer honnetement le modele.
  2. Overfitting : avec seulement ~20 exemples, le risque est eleve que le
     modele "apprenne par coeur" au lieu de generaliser.

Usage :
    python evaluer_modele.py
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from classifier import TRAINING_DATA, _build_pipeline

# On evalue ici uniquement la cible "categorie", le principe est identique
# pour "gravite" et "departement".
textes = [t for t, _, _, _ in TRAINING_DATA]
labels = [c for _, c, _, _ in TRAINING_DATA]

# ---------------------------------------------------------------------
# TRAIN/TEST SPLIT
# On separe les donnees en 2 groupes :
#   - train (ex: 70%) : utilise pour entrainer le modele
#   - test  (ex: 30%) : jamais vu pendant l'entrainement, sert a verifier
#     que le modele generalise et ne fait pas que "recracher" ce qu'il a
#     appris par coeur.
# ---------------------------------------------------------------------
try:
    X_train, X_test, y_train, y_test = train_test_split(
        textes, labels,
        test_size=0.3,      # 30% des donnees mises de cote pour le test
        random_state=42,     # pour que le decoupage soit reproductible
        stratify=labels,      # garde les memes proportions de chaque categorie
    )
except ValueError as e:
    # Cette erreur est elle-meme une lecon : certaines categories n'ont
    # qu'1 seul exemple dans TRAINING_DATA, impossible de les repartir
    # entre train et test. C'est un signe direct de dataset trop petit.
    print(f"! stratify impossible ({e})")
    print("! -> certaines categories ont trop peu d'exemples pour etre separees")
    print("!    en train/test. On continue sans stratify, a titre pedagogique.\n")
    X_train, X_test, y_train, y_test = train_test_split(
        textes, labels, test_size=0.3, random_state=42,
    )

print(f"Total : {len(textes)} exemples -> {len(X_train)} pour l'entrainement, {len(X_test)} pour le test\n")

# Entrainement UNIQUEMENT sur X_train (pas sur toutes les donnees)
pipeline = _build_pipeline()
pipeline.fit(X_train, y_train)

# Score sur les donnees d'entrainement (ce que le modele a deja "vu")
score_train = accuracy_score(y_train, pipeline.predict(X_train))

# Score sur les donnees de test (jamais vues -> le vrai indicateur de qualite)
score_test = accuracy_score(y_test, pipeline.predict(X_test))

print(f"Precision sur les donnees d'ENTRAINEMENT : {score_train:.0%}")
print(f"Precision sur les donnees de TEST        : {score_test:.0%}")

# ---------------------------------------------------------------------
# OVERFITTING
# Si le score train est tres eleve (souvent 100% avec si peu de donnees)
# mais que le score test est nettement plus bas, c'est le signe classique
# d'overfitting : le modele a memorise les exemples d'entrainement au lieu
# d'apprendre des regles generales.
# ---------------------------------------------------------------------
ecart = score_train - score_test
print(f"\nEcart train/test : {ecart:.0%}")
if ecart > 0.2:
    print("-> Ecart important : signe d'overfitting probable.")
    print("   Cause la plus probable ici : trop peu d'exemples d'entrainement")
    print("   (~20 au total, donc ~14 par le split, pour 5 categories).")
    print("   Solution : ajouter beaucoup plus d'exemples reels des que possible.")
else:
    print("-> Ecart raisonnable, mais reste prudent avec un jeu de donnees aussi petit.")

print("\nExemples mal predits sur le jeu de test :")
predictions_test = pipeline.predict(X_test)
for texte, vrai, predit in zip(X_test, y_test, predictions_test):
    marqueur = "OK" if vrai == predit else "ERREUR"
    print(f"  [{marqueur}] \"{texte}\" -> predit: {predit} | attendu: {vrai}")
