#!/usr/bin/env python3
"""Assemble the seminar page from src/.

  python3 build.py                      -> index.html (the published page)
  python3 build.py --only vae --out work/vae.html
        uses src/regions/vae for that region and the frozen src/baseline/regions/* for all others,
        so parallel work on other regions cannot break your test page.

Region = src/regions/<name>/{index.html, style.css, script.js}. Placeholders {{name}} live in src/page.html.
"""
import argparse, os, re, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'src')
ORDER = ['chrome', 'hero', 'vae', 'gan', 'dif', 'tri', 'time', 'pick', 'end']

def read(p):
    with open(p, encoding='utf-8') as f: return f.read()

def region_dir(name, only):
    if only and name not in only:
        b = os.path.join(SRC, 'baseline', 'regions', name)
        if os.path.isdir(b): return b
    return os.path.join(SRC, 'regions', name)

def build(only=None):
    page = read(os.path.join(SRC, 'page.html'))
    css = [read(os.path.join(SRC, 'base.css'))]
    js = ['<script>\n' + read(os.path.join(SRC, 'core.js')) + '</script>']
    for name in ORDER:
        d = region_dir(name, only)
        html = read(os.path.join(d, 'index.html'))
        page = page.replace('{{' + name + '}}', html.rstrip('\n'))
        c = read(os.path.join(d, 'style.css')).strip()
        if c: css.append(f'/* ===== region: {name} ===== */\n' + c)
        j = read(os.path.join(d, 'script.js')).strip()
        if j: js.append(f'<script>\n/* ===== region: {name} ===== */\n' + j + '\n</script>')
    left = re.findall(r'\{\{\w+\}\}', page)
    if left: sys.exit(f'unfilled placeholders: {left}')
    frag = read(os.path.join(SRC, 'head.html')) + '<style>\n' + '\n\n'.join(css) + '\n</style>\n\n' + page + '\n' + '\n'.join(js) + '\n'
    full = ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">' + frag + '</html>')
    return frag, full

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', action='append', help='region(s) to take from src/regions; others from baseline')
    ap.add_argument('--out', help='write the page here instead of index.html')
    a = ap.parse_args()
    frag, full = build(a.only)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, 'w', encoding='utf-8').write(full)
        print(a.out)
    else:
        open(os.path.join(ROOT, 'index.html'), 'w', encoding='utf-8').write(full)
        print('index.html')
