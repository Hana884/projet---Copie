import sys
import nbformat
from nbclient import NotebookClient

print("Démarrage de l'exécution du notebook...")
with open("Mini_Projet_Series_Temporelles.ipynb", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")
try:
    client.execute()
    with open("Mini_Projet_Series_Temporelles.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Succès : Toutes les cellules du notebook ont été exécutées et les graphiques/sorties sont pré-calculés !")
except Exception as e:
    print("Erreur d'exécution :", e)
    sys.exit(1)
