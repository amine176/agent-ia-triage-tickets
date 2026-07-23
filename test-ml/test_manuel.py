"""
test_manuel.py
--------------
Petit script pour tester classifier.py et similarity.py avec tes propres
phrases, sans avoir a modifier le code source.
Usage :
    python classifier.py          # une seule fois, pour entrainer les modeles
    python test_manuel.py
"""

from classifier import predict
from similarity import trouver_ticket_similaire, rattacher_ou_creer_groupe, TicketOuvert

print("=" * 70)
print("TEST 1 - Classification (classifier.py)")
print("=" * 70)

mes_tickets = [
    ("L'imprimante du 3eme etage ne repond plus", "Collaborateur"),
    ("Je n'ai pas recu mon solde de tout compte", "Manager"),
    ("Le site web de l'entreprise est inaccessible depuis ce matin", "Directeur"),
]

for texte, poste in mes_tickets:
    resultat = predict(texte, poste=poste)
    print(f"\nTexte    : {texte}")
    print(f"Poste    : {poste}")
    print(f"Resultat : {resultat}")


print("\n" + "=" * 70)
print("TEST 2 - Detection de doublons (similarity.py)")
print("=" * 70)

tickets_ouverts = [
    TicketOuvert(1, "Panne imprimante", "L'imprimante du 3eme etage est en panne depuis hier", "Incident Services Generaux", "Services Generaux"),
    TicketOuvert(2, "Wifi coupe", "Le wifi ne fonctionne plus dans la salle de reunion", "Incident IT", "DSI"),
]

nouveau_texte = "L'imprimante du 3eme etage ne repond plus du tout"
resultat = predict(nouveau_texte, poste="Collaborateur")

match = trouver_ticket_similaire(
    nouveau_texte,
    resultat["categorie"],
    resultat["departement"],
    tickets_ouverts,
)

print(f"\nNouveau texte : {nouveau_texte}")
print(f"Classifie comme : {resultat['categorie']} / {resultat['departement']}")
print(f"Ticket similaire trouve : {match}")
print(f"Action a prendre : {rattacher_ou_creer_groupe(match, nouveau_ticket_id=99)}")
