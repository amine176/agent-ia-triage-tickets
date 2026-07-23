"""
similarity.py
--------------
Detection des demandes similaires (doublons), pour eviter de traiter
plusieurs fois le meme probleme signale par des collaborateurs differents
(cf. note de cadrage : table ProblemeGroupes / ProblemeUtilisateurs).

Approche : similarite cosinus sur des embeddings de phrases (modele
multilingue "paraphrase-multilingual-MiniLM-L12-v2"), restreinte aux
tickets ouverts de la meme categorie et/ou du meme departement (le texte
seul ne suffit pas : deux tickets tres proches textuellement mais dans des
departements differents ne doivent pas etre regroupes).

Les embeddings capturent le sens plutot que les mots exacts : "le VPN ne
se connecte plus" et "impossible de me connecter au VPN" sont ainsi
reconnus comme similaires malgre une formulation differente, ce qu'un
simple TF-IDF ne detecte pas (~0.18 de similarite au lieu de ~0.75+ ici).

Si sentence-transformers n'est pas installe ou indisponible (pas de reseau
pour telecharger le modele au premier lancement), le module retombe
automatiquement sur une similarite TF-IDF (moins precise mais 100% locale).
"""

from dataclasses import dataclass
from typing import List, Optional
import re
import unicodedata

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SEUIL_SIMILARITE = 0.65  # a ajuster selon les faux positifs/negatifs observes
MODELE_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"

_model = None
_mode = None  # "embeddings" ou "tfidf"


def _get_model():
    """Charge le modele d'embeddings une seule fois (lazy loading)."""
    global _model, _mode
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODELE_EMBEDDINGS)
        _mode = "embeddings"
    except Exception:
        # Pas de reseau / package absent -> repli sur TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        _model = TfidfVectorizer(preprocessor=_normalize, ngram_range=(1, 2), min_df=1)
        _mode = "tfidf"
    return _model


@dataclass
class TicketOuvert:
    id: int
    titre: str
    description: str
    categorie: str
    departement: str
    groupe_id: Optional[int] = None  # id du ProblemeGroupe si deja rattache


def _normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def trouver_ticket_similaire(
    nouveau_texte: str,
    nouvelle_categorie: str,
    nouveau_departement: str,
    tickets_ouverts: List[TicketOuvert],
    seuil: float = SEUIL_SIMILARITE,
) -> Optional[TicketOuvert]:
    """Compare le nouveau ticket aux tickets ouverts du meme departement et
    de la meme categorie. Retourne le ticket le plus proche au-dessus du
    seuil, ou None si aucun match."""

    # 1. Filtrage metier : meme departement (obligatoire) + meme categorie
    #    (assouplissable si on veut detecter des doublons inter-categories)
    candidats = [
        t for t in tickets_ouverts
        if t.departement == nouveau_departement and t.categorie == nouvelle_categorie
    ]
    if not candidats:
        return None

    # 2. Similarite semantique (embeddings) sur titre + description
    textes_candidats = [f"{t.titre} {t.description}" for t in candidats]
    corpus = [nouveau_texte] + textes_candidats

    model = _get_model()
    if _mode == "embeddings":
        vecteurs = model.encode(corpus, normalize_embeddings=True)
        scores = cosine_similarity([vecteurs[0]], vecteurs[1:]).flatten()
    else:  # repli TF-IDF
        matrice = model.fit_transform(corpus)
        scores = cosine_similarity(matrice[0:1], matrice[1:]).flatten()

    meilleur_index = int(np.argmax(scores))
    meilleur_score = scores[meilleur_index]

    if meilleur_score >= seuil:
        return candidats[meilleur_index]
    return None


def rattacher_ou_creer_groupe(ticket_similaire: Optional[TicketOuvert], nouveau_ticket_id: int):
    """Logique d'integration avec les tables ProblemeGroupes / ProblemeUtilisateurs.
    A brancher sur les vraies fonctions SQLAlchemy/SQLite de l'application."""
    if ticket_similaire is None:
        # Aucun doublon détecté : le ticket est traité isolément.
        return {"action": "ticket_isole", "ticket_id": nouveau_ticket_id}

    if ticket_similaire.groupe_id is not None:
        # Le ticket-maitre appartient deja a un groupe : on y ajoute l'utilisateur/ticket.
        return {
            "action": "ajout_au_groupe_existant",
            "groupe_id": ticket_similaire.groupe_id,
            "ticket_maitre_id": ticket_similaire.id,
            "nouveau_ticket_id": nouveau_ticket_id,
        }

    # Aucun groupe encore cree pour ce ticket-maitre : on en cree un.
    return {
        "action": "creation_nouveau_groupe",
        "ticket_maitre_id": ticket_similaire.id,
        "nouveau_ticket_id": nouveau_ticket_id,
    }


if __name__ == "__main__":
    tickets_ouverts = [
        TicketOuvert(1, "VPN inaccessible", "Le VPN ne se connecte plus depuis ce matin", "Incident IT", "DSI"),
        TicketOuvert(2, "Probleme imprimante", "L'imprimante du 2eme etage n'imprime plus", "Incident Services Generaux", "Services Generaux"),
    ]

    nouveau = "Impossible de me connecter au VPN de l'entreprise depuis ce matin"
    match = trouver_ticket_similaire(nouveau, "Incident IT", "DSI", tickets_ouverts)
    print("Ticket similaire trouve :", match)
    print(rattacher_ou_creer_groupe(match, nouveau_ticket_id=3))
