"""Check JOSS source structure, citations, figures and benchmark provenance."""
import hashlib
import json
from pathlib import Path
import re

import bibtexparser
import yaml

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
text = (PAPER / 'paper.md').read_text()
_, metadata, body = text.split('---', 2)
meta = yaml.safe_load(metadata)
for key in ['title', 'authors', 'affiliations', 'date', 'bibliography']:
    assert key in meta, f'Missing metadata: {key}'
for heading in ['Summary', 'Statement of need', 'State of the field',
                'Software design', 'Research impact statement',
                'AI usage disclosure', 'Acknowledgements', 'References']:
    assert f'# {heading}\n' in body, f'Missing section: {heading}'
body = re.sub(r'<!--.*?-->', '', body, flags=re.S)
with (PAPER / meta['bibliography']).open() as handle:
    references = bibtexparser.load(handle)
keys = {entry['ID'] for entry in references.entries}
citations = set(re.findall(r'@([A-Za-z][A-Za-z0-9_]*(?:[:-][A-Za-z0-9_]+)*)', body))
assert citations <= keys, f'Undefined citations: {citations - keys}'
for entry in references.entries:
    assert 'doi' in entry or 'url' in entry, f'Missing locator: {entry["ID"]}'
for image in re.findall(r'!\[.*?\]\(([^)]+)\)', body, flags=re.S):
    assert (PAPER / image).is_file(), f'Missing figure: {image}'
labels = set(re.findall(r'\\label\{([^}]+)\}', body))
refs = set(re.findall(r'\\(?:auto)?ref\{([^}]+)\}', body))
assert refs <= labels, f'Undefined cross-references: {refs - labels}'
words = len(re.findall(r"\b[\w'-]+\b", re.sub(r'\{[^}]*\}|\([^)]*\.png\)', '', body)))
assert 750 <= words <= 1750, f'JOSS word-range check failed: {words}'
results = json.loads((PAPER / 'benchmarks/results.json').read_text())
for filename, expected in results['metadata']['source_sha256'].items():
    assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, (
        f'Benchmark source changed; regenerate measurements: {filename}')
assert len(results['rows']) == 72
assert all(len(row['seconds_per_call']) == 7 for row in results['rows'])
print(f'Passed: metadata, sections, {len(citations)} citations, '
      f'{len(labels)} figure references, {words} approximate words, source hashes.')
print('Author confirmation of scientific impact, funding and AI review remains required.')
