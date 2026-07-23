"""
classifier.py
--------------
Classification automatique des demandes/reclamations : categorie, gravite,
departement cible, puis calcul de la priorite finale (gravite + poste du
demandeur), conformement a la note de cadrage du projet.

Approche : TF-IDF + regression logistique (scikit-learn), entrainee ici sur
un jeu de donnees synthetique construit a partir des regles metier
(mots-cles -> departement/categorie). Remplace TRAINING_DATA par de vrais
tickets historiques des qu'ils sont disponibles pour ameliorer la precision.

Utilisation :
    from classifier import train, predict

    train()  # a lancer une fois (ou apres mise a jour des donnees)
    resultat = predict("Mon ordinateur ne demarre plus", poste="Manager")
    # -> {'categorie': 'Incident IT', 'gravite': 'Moyenne',
    #     'departement': 'DSI', 'priorite': 'Haute'}
"""

import os
import re
import unicodedata
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Jeu de donnees d'entrainement (synthetique -> a remplacer par des tickets
#    reels au fil du temps). Chaque ligne : (texte, categorie, gravite, departement)
# ---------------------------------------------------------------------------
TRAINING_DATA = [
    ("Mon ordinateur ne demarre plus depuis ce matin", "Incident IT", "Moyenne", "DSI"),
    ("Impossible de me connecter au reseau wifi du bureau", "Incident IT", "Faible", "DSI"),
    ("L'application de facturation plante a chaque ouverture", "Incident IT", "Elevee", "DSI"),
    ("Le serveur de messagerie est completement hors service", "Incident IT", "Critique", "DSI"),
    ("Ecran bleu et redemarrage impossible sur mon poste", "Incident IT", "Elevee", "DSI"),
    ("Le VPN ne se connecte plus depuis la mise a jour", "Incident IT", "Moyenne", "DSI"),

    ("Je souhaite poser une semaine de conge en aout", "Demande RH", "Faible", "RH"),
    ("Ma fiche de paie de ce mois comporte une erreur", "Reclamation RH", "Moyenne", "RH"),
    ("Suivi de mon dossier de recrutement pour le poste ouvert", "Demande RH", "Faible", "RH"),
    ("Mon solde de conges affiche est incorrect", "Reclamation RH", "Moyenne", "RH"),
    ("Demande d'attestation de travail pour mon dossier", "Demande RH", "Faible", "RH"),

    ("La facture du fournisseur n'a toujours pas ete payee", "Demande Finance", "Moyenne", "Finance"),
    ("Depassement du budget alloue au projet ce trimestre", "Reclamation Finance", "Elevee", "Finance"),
    ("Le remboursement de mes frais de deplacement est en retard", "Demande Finance", "Faible", "Finance"),
    ("Erreur sur le montant facture par le prestataire", "Reclamation Finance", "Moyenne", "Finance"),

    ("Le contrat fournisseur n'est pas conforme aux clauses convenues", "Reclamation Juridique", "Elevee", "Juridique"),
    ("Question sur la conformite RGPD de notre nouveau formulaire", "Demande Juridique", "Moyenne", "Juridique"),
    ("Revision necessaire d'une clause du contrat de partenariat", "Demande Juridique", "Moyenne", "Juridique"),

    ("Mon bureau n'a plus de chaise et le materiel est casse", "Demande Services Generaux", "Faible", "Services Generaux"),
    ("La climatisation de la salle de reunion ne fonctionne plus", "Incident Services Generaux", "Moyenne", "Services Generaux"),
    ("Besoin de nouvelles fournitures de bureau pour l'equipe", "Demande Services Generaux", "Faible", "Services Generaux"),
]

# Ponderation gravite -> score numerique (utilise pour le calcul de priorite finale)
GRAVITE_SCORE = {"Faible": 1, "Moyenne": 2, "Elevee": 3, "Critique": 4}

# Ponderation poste -> impact sur la priorite (cf. note de cadrage, section 3)
POSTE_SCORE = {
    "Directeur": 4,          # Tres eleve
    "Chef de departement": 3,  # Eleve
    "Manager": 2,             # Moyen
    "Collaborateur": 1,       # Standard
}

PRIORITE_LABELS = ["Basse", "Moyenne", "Haute", "Critique"]


def _normalize(text: str) -> str:
    """Minuscule + suppression des accents, pour un TF-IDF plus robuste."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=_normalize, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train(training_data=None):
    """Entraine 3 classifieurs independants (categorie, gravite, departement)
    et les sauvegarde dans MODEL_DIR."""
    data = training_data or TRAINING_DATA
    textes = [t for t, _, _, _ in data]
    targets = {
        "categorie": [c for _, c, _, _ in data],
        "gravite": [g for _, _, g, _ in data],
        "departement": [d for _, _, _, d in data],
    }

    for nom_cible, y in targets.items():
        pipeline = _build_pipeline()
        pipeline.fit(textes, y)
        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{nom_cible}.joblib"))

    print(f"Modeles entraines sur {len(textes)} exemples et sauvegardes dans {MODEL_DIR}/")


def _load_model(nom_cible):
    path = os.path.join(MODEL_DIR, f"{nom_cible}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modele '{nom_cible}' introuvable. Lance classifier.train() d'abord."
        )
    return joblib.load(path)


def _priorite_depuis_scores(gravite_score: int, poste_score: int) -> str:
    """Combine gravite (poids 2) et poste du demandeur (poids 1) pour deriver
    une priorite finale sur 4 niveaux."""
    score_total = gravite_score * 2 + poste_score  # echelle 3 a 12
    if score_total >= 10:
        return "Critique"
    if score_total >= 7:
        return "Haute"
    if score_total >= 4:
        return "Moyenne"
    return "Basse"


def predict(texte: str, poste: str = "Collaborateur") -> dict:
    """Retourne la categorie, la gravite, le departement cible et la
    priorite finale (ajustee selon le poste du demandeur) pour un ticket."""
    categorie = _load_model("categorie").predict([texte])[0]
    gravite = _load_model("gravite").predict([texte])[0]
    departement = _load_model("departement").predict([texte])[0]

    gravite_score = GRAVITE_SCORE.get(gravite, 2)
    poste_score = POSTE_SCORE.get(poste, 1)
    priorite = _priorite_depuis_scores(gravite_score, poste_score)

    return {
        "categorie": categorie,
        "gravite": gravite,
        "departement": departement,
        "priorite": priorite,
    }


if __name__ == "__main__":
    train()
    exemples = [
        ("Mon ordinateur ne demarre plus", "Manager"),
        ("Mon ordinateur ne demarre plus", "Directeur"),
        ("Je voudrais poser un conge la semaine prochaine", "Collaborateur"),
        ("Le serveur de facturation est completement bloque", "Chef de departement"),
    ]
    for texte, poste in exemples:
        print(texte, "|", poste, "->", predict(texte, poste))
