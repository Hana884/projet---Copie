import json
from pathlib import Path
nb = json.load(open('Mini_Projet_Series_Temporelles_execute.ipynb', encoding='utf-8'))
outdir = Path('results/tables')
n = 0
for i, cell in enumerate(nb.get('cells', []), 1):
    for j, out in enumerate(cell.get('outputs', []), 1):
        data = out.get('data', {})
        html = data.get('text/html', '')
        if isinstance(html, list): html = ''.join(html)
        if html and '<table' in html.lower():
            (outdir / f'table_cell_{i}_{j}.html').write_text(html, encoding='utf-8')
            n += 1
        txt = data.get('text/plain', '')
        if isinstance(txt, list): txt = ''.join(txt)
        if txt.strip():
            (outdir / f'resultat_cell_{i}_{j}.txt').write_text(txt, encoding='utf-8')
            n += 1
        if out.get('output_type') == 'stream':
            txt = out.get('text', '')
            if isinstance(txt, list): txt = ''.join(txt)
            if txt.strip():
                (outdir / f'console_cell_{i}_{j}.txt').write_text(txt, encoding='utf-8')
                n += 1
print('Fichiers de résultats extraits :', n)
print('Dossier :', outdir.resolve())
