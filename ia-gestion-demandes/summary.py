import os
import json
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

SYSTEM_PROMPT_RESUME = """Tu es un assistant qui rédige des synthèses hebdomadaires pour des responsables d'entreprise.
À partir d'une liste de tickets (incidents/réclamations), rédige un résumé clair et concis en français, structuré ainsi :
1. Volume de demandes par département
2. Tickets critiques en cours
3. Tendances observées

Reste factuel et synthétique, en 150 mots maximum.
"""


def generer_resume(tickets: list) -> str:
    tickets_texte = "\n".join([
        f"- [{t['departement']}] {t['titre']} (gravité: {t['gravite']}, statut: {t['statut']})"
        for t in tickets
    ])

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_RESUME},
            {"role": "user", "content": f"Voici les tickets de la semaine :\n{tickets_texte}"}
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    tickets_semaine = [
        {"titre": "PC ne démarre plus", "departement": "DSI", "gravite": "Moyenne", "statut": "Ouvert"},
        {"titre": "Serveur mail en panne", "departement": "DSI", "gravite": "Critique", "statut": "Ouvert"},
        {"titre": "Demande de congé", "departement": "RH", "gravite": "Faible", "statut": "Fermé"},
        {"titre": "Facture impayée fournisseur", "departement": "Finance", "gravite": "Élevée", "statut": "Ouvert"},
        {"titre": "Réseau lent", "departement": "DSI", "gravite": "Moyenne", "statut": "Fermé"},
    ]

    resume = generer_resume(tickets_semaine)
    print(resume)