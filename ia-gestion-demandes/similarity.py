import os
import json
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

SYSTEM_PROMPT_SIMILARITE = """Tu compares deux tickets internes d'entreprise pour savoir s'ils décrivent le même problème.
Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ou après.
Format attendu :
{"meme_probleme": true} ou {"meme_probleme": false}
"""


def sont_similaires(ticket_a: dict, ticket_b: dict) -> bool:
    contenu = f"""Ticket 1:
Titre: {ticket_a['titre']}
Description: {ticket_a['description']}

Ticket 2:
Titre: {ticket_b['titre']}
Description: {ticket_b['description']}
"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SIMILARITE},
            {"role": "user", "content": contenu}
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("meme_probleme", False)


def trouver_ticket_similaire(nouveau_ticket: dict, tickets_existants: list) -> dict:
    for ticket in tickets_existants:
        if ticket.get("statut") == "Fermé":
            continue
        if ticket.get("categorie") != nouveau_ticket.get("categorie"):
            continue
        if ticket.get("departement") != nouveau_ticket.get("departement"):
            continue

        if sont_similaires(nouveau_ticket, ticket):
            return {"trouve": True, "ticket_id": ticket["id"]}

    return {"trouve": False}


if __name__ == "__main__":
    tickets_existants = [
        {
            "id": 1,
            "titre": "PC ne démarre plus",
            "description": "Écran noir depuis ce matin, impossible de travailler",
            "categorie": "Incident IT",
            "departement": "DSI",
            "statut": "Ouvert"
        },
        {
            "id": 2,
            "titre": "Demande de congé",
            "description": "Je souhaite poser une semaine de congé en août",
            "categorie": "Demande RH",
            "departement": "RH",
            "statut": "Ouvert"
        }
    ]

    nouveau_ticket = {
        "titre": "Mon ordinateur ne s'allume pas",
        "description": "Depuis ce matin l'écran reste noir, je ne peux pas travailler",
        "categorie": "Incident IT",
        "departement": "DSI"
    }

    resultat = trouver_ticket_similaire(nouveau_ticket, tickets_existants)
    print(resultat)