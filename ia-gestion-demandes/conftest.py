import os
import sys

# Permet d'importer app.py, classifier.py, similarity.py, summary.py
# depuis le dossier tests/ (qui est un sous-dossier du projet).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Clé factice pour que les modules puissent s'importer sans .env réel.
os.environ.setdefault("MISTRAL_API_KEY", "cle_de_test_fictive")
