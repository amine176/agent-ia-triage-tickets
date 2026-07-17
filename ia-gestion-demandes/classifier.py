import os
import json
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

IMPACT_POSTE = {
    "Directeur": "Très élevé",
    "Chef de département": "Élevé",
    "Manager": "Moyen",
    "Collaborateur": "Standard"
}

SYSTEM_PROMPT = """Tu es un assistant qui analyse des demandes/réclamations internes d'une entreprise.
Pour chaque demande, tu dois déterminer :
- categorie : ex. "Incident IT", "Demande RH", "Demande Finance", "Demande Juridique", "Demande Services Généraux"
- gravite : "Faible", "Moyenne", "Élevée", "Critique"
- priorite : "Basse", "Moyenne", "Haute", "Urgente"
- departement : "DSI", "RH", "Finance", "Juridique", "Services Généraux"

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ou après, sans balises markdown.
Format attendu :
{"categorie": "...", "gravite": "...", "priorite": "...", "departement": "..."}
"""

def classify_ticket(titre: str, description: str, poste: str = None) -> dict:
    user_content = f"Titre: {titre}\nDescription: {description}"

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    if poste and poste in IMPACT_POSTE:
        result["impact_poste"] = IMPACT_POSTE[poste]

    return result


if __name__ == "__main__":
    test = classify_ticket(
        titre="PC ne démarre plus",
        description="Écran noir depuis ce matin, impossible de travailler",
        poste="Manager"
    )
    print(json.dumps(test, indent=2, ensure_ascii=False))