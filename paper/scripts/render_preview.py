"""Render a reading PDF locally; the official JOSS proof uses Inara.

Requires markdown, PyYAML, bibtexparser, matplotlib and wkhtmltopdf.
The local preview is deliberately labelled as a draft, not a journal proof.
"""
import html
from pathlib import Path
import re
import shutil
import subprocess

import bibtexparser
from bibtexparser.customization import convert_to_unicode
import markdown
import matplotlib
matplotlib.use('Agg')
from matplotlib.mathtext import math_to_image
import yaml

PAPER = Path(__file__).resolve().parents[1]
OUTPUT = PAPER / 'output'
OUTPUT.mkdir(exist_ok=True)
_, metadata, body = (PAPER / 'paper.md').read_text().split('---', 2)
meta = yaml.safe_load(metadata)
parser = bibtexparser.bparser.BibTexParser(common_strings=True)
parser.customization = convert_to_unicode
with (PAPER / meta['bibliography']).open() as handle:
    entries = bibtexparser.load(handle, parser=parser).entries
lookup = {e['ID']: e for e in entries}


def author_label(entry):
    if entry['ID'] == 'Astropy2022':
        return 'Astropy Collaboration'
    if entry['ID'] == 'Lightkurve2018':
        return 'Lightkurve Collaboration'
    authors = entry['author'].split(' and ')
    surnames = [author.split(',')[0].strip('{}') for author in authors]
    if len(surnames) == 1:
        return surnames[0]
    if len(surnames) == 2:
        return ' & '.join(surnames)
    return surnames[0] + ' et al.'


# Disambiguate same-author, same-year citations in the local reading copy.
# The official JOSS build performs its own citeproc disambiguation.
groups = {}
for entry in entries:
    groups.setdefault((author_label(entry), entry['year']), []).append(entry)
display_year = {entry['ID']: entry['year'] for entry in entries}
for group in groups.values():
    if len(group) > 1:
        for index, entry in enumerate(sorted(group, key=lambda e: e['title'])):
            display_year[entry['ID']] += chr(ord('a') + index)

body = re.sub(r'<!--.*?-->', '', body, flags=re.S)
# Display equations are rendered locally, avoiding any network-loaded MathJax.
count = 0

def equation(match):
    global count
    count += 1
    path = OUTPUT / f'equation-{count}.png'
    math_to_image('$' + match[1].strip() + '$', str(path), dpi=180, format='png')
    return f'<div class="equation"><img src="{path.as_uri()}" /></div>'


body = re.sub(r'\$\$(.*?)\$\$', equation, body, flags=re.S)
inline_math = {'\\mu': 'μ', '\\alpha': 'α', 'e=0.4': 'e = 0.4', 'P_n=(t_2-t_1)/n': 'Pₙ = (t₂ − t₁)/n'}
body = re.sub(r'\$([^$]+)\$', lambda m: inline_math.get(m[1], m[1]), body)
figure_numbers = {}


def figure(match):
    caption, filename = match[1], match[2]
    label = re.search(r'\\label\{([^}]+)\}', caption)
    number = len(figure_numbers) + 1
    if label:
        figure_numbers[label[1]] = number
    caption = re.sub(r'\\label\{[^}]+\}', '', caption)
    return (f'<figure><img src="{(PAPER / filename).as_uri()}" />'
            f'<figcaption><b>Figure {number}.</b> {html.escape(caption)}</figcaption></figure>')


body = re.sub(r'!\[(.*?)\]\(([^)]+)\)\{[^}]*\}', figure, body, flags=re.S)
body = re.sub(r'\\ref\{([^}]+)\}', lambda m: str(figure_numbers[m[1]]), body)
def parenthetical(match):
    keys = re.findall(r'@([A-Za-z0-9]+)', match[1])
    return '(' + '; '.join(author_label(lookup[key]) + ', ' + display_year[key]
                            for key in keys) + ')'


body = re.sub(r'\[([^\[\]\n]*@[^\[\]\n]*)\]', parenthetical, body)
body = re.sub(r'@([A-Za-z0-9]+)',
              lambda m: author_label(lookup[m[1]]) + ' (' + display_year[m[1]] + ')', body)
references = []
for entry in sorted(entries, key=lambda e: (author_label(e), e['year'])):
    authors = entry['author'].split(' and ')
    author_text = '; '.join(authors[:3]) + ('; et al.' if len(authors) > 3 else '')
    venue = entry.get('journal', entry.get('howpublished', ''))
    volume = entry.get('volume', '')
    pages = entry.get('pages', '').replace('--', '-')
    locator = ('https://doi.org/' + entry['doi']) if 'doi' in entry else entry['url']
    text = (f'{author_text} ({display_year[entry["ID"]]}). {entry["title"]}. '
            f'{venue}, {volume}, {pages}.')
    references.append('<p>' + html.escape(text) + ' <a href="' + html.escape(locator) + '">' + html.escape(locator) + '</a></p>')
body = body.replace('# References', '# References\n\n<div class="references">' + '\n'.join(references) + '</div>')
rendered = markdown.markdown(body, extensions=['extra'])
css = '''
body { font-family: Georgia, 'DejaVu Serif', serif; color: #20252b;
       font-size: 10.5pt; line-height: 1.28; }
h1 { font-size: 13pt; margin: 18px 0 7px; page-break-after: avoid; }
h2 { font-size: 11.5pt; margin: 14px 0 6px; page-break-after: avoid; }
p { margin: 0 0 9px; orphans: 3; widows: 3; }
a { color: #185b87; text-decoration: none; }
.title { font-size: 20pt; line-height: 1.15; color: #174d69; margin: 12px 0; }
.author { font-size: 12pt; margin-bottom: 3px; }
.affiliation, .meta { font-size: 9pt; color: #59656b; }
.draft { font-family: sans-serif; font-size: 8pt; letter-spacing: .6px;
         color: #59656b; border-bottom: 1px solid #a9bbc4; padding-bottom: 7px; }
figure { margin: 14px 0; page-break-inside: avoid; }
figure img { display: block; width: 92%; margin: 0 auto; }
figcaption { font-size: 9pt; line-height: 1.25; margin-top: 6px; }
.equation { text-align: center; padding: 7px 0 12px; }
.equation img { height: 25px; width: auto; }
.references { font-size: 9pt; line-height: 1.25; }
.references p { margin-bottom: 8px; page-break-inside: avoid; }
code { font-family: monospace; font-size: 9pt; }
'''
author_names = ', '.join(author['name'] for author in meta['authors'])
affiliations = '<br>'.join(html.escape(item['name']) for item in meta['affiliations'])
orcids = ', '.join(author['orcid'] for author in meta['authors'] if author.get('orcid'))
header = (f'<div class="draft">BRUCE | AUTHOR DRAFT | LOCAL READING PREVIEW</div>'
          f'<div class="title">{html.escape(meta["title"])}</div>'
          f'<div class="author">{html.escape(author_names)}</div>'
          f'<div class="affiliation">{affiliations}<br>'
          f'ORCID: {html.escape(orcids)}</div>'
          f'<p class="meta">{meta["date"]} · JOSS manuscript draft · Not an official journal proof</p>')
path = OUTPUT / 'paper-preview.html'
path.write_text('<!doctype html><html><head><meta charset="UTF-8"><style>' + css +
                '</style></head><body>' + header + rendered + '</body></html>')
program = shutil.which('wkhtmltopdf')
if not program:
    raise SystemExit('HTML generated. Install wkhtmltopdf for the local PDF preview, or use Inara.')
subprocess.run([program, '--enable-local-file-access', '--page-size', 'A4',
                '--margin-top', '17mm', '--margin-bottom', '18mm',
                '--margin-left', '20mm', '--margin-right', '20mm',
                '--footer-center', '[page] / [topage]', '--footer-font-size', '8',
                str(path), str(OUTPUT / 'paper-preview.pdf')], check=True)
print(OUTPUT / 'paper-preview.pdf')
