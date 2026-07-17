# Agent IA de triage des demandes

Application web (Flask + API Mistral) qui classe automatiquement des demandes/réclamations internes d'entreprise, détecte les tickets similaires (doublons) et génère un résumé hebdomadaire.

## Fonctionnalités

- **Classification automatique** : catégorie, gravité, priorité, département (`classifier.py`)
- **Détection de doublons** : comparaison avec les tickets existants (`similarity.py`)
- **Résumé hebdomadaire** : synthèse générée par l'IA (`summary.py`)
- **Interface web** : console de type chat (`app.py`)

## Architecture

| Fichier | Rôle |
|---|---|
| `app.py` | Serveur Flask, interface web, routes API |
| `classifier.py` | Classification d'un ticket via Mistral |
| `similarity.py` | Détection de tickets similaires |
| `summary.py` | Génération du résumé hebdomadaire |

## Installation

```bash
git clone <url-du-repo>
cd <nom-du-repo>
python -m venv venv
source venv/bin/activate   # sous Windows : venv\Scripts\activate
pip install flask mistralai python-dotenv
```

Créer un fichier `.env` à la racine avec votre clé API :
