import json, base64
from pathlib import Path
nb = json.load(open('Mini_Projet_Series_Temporelles_execute.ipynb', encoding='utf-8'))
fig = Path('results/figures')
tab = Path('results/tables')
nimg = 0
ntab = 0
for i, cell in enumerate(nb.get('cells', []), 1):
    for j, out in enumerate(cell.get('outputs', []), 1):
        data = out.get('data', {})
        if 'image/png' in data:
            raw = data['image/png']
            if isinstance(raw, list): raw = ''.join(raw)
            (fig / f'figure_cell_{i}_{j}.png').write_bytes(base64.b64decode(raw))
            nimg += 1
        txt = data.get('text/plain', '')
        if isinstance(txt, list): txt = ''.join(txt)
        if txt.strip() and len(txt) < 100000 and ('DataFrame' in txt or 'MSE' in txt or 'RMSE' in txt or 'Statistiques' in txt or 'Variable' in txt):
            (tab / f'resultat_cell_{i}_{j}.txt').write_text(txt, encoding='utf-8')
            ntab += 1
print('Graphiques extraits :', nimg)
print('Résultats texte extraits :', ntab)
print('Figures :', fig.resolve())
print('Tables :', tab.resolve())
