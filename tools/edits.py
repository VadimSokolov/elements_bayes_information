#!/usr/bin/env python3
"""Apply exact-text edits to the book source, safely, even with several agents at once.

Edit files use a plain-text format that needs no escaping, so LaTeX backslashes are safe:

    @@@ EDIT <id> | <optional tags>
    @@@ WHY <optional one-line rationale>
    @@@ OLD
    <the exact current text, any number of lines>
    @@@ NEW
    <the replacement, any number of lines; empty deletes the OLD text>
    @@@ END

Usage:
    python3 tools/edits.py check EDITS [--source FILE]   # simulate, change nothing
    python3 tools/edits.py apply EDITS [--source FILE]   # apply under an exclusive lock

OLD is matched verbatim first; if that fails, runs of whitespace are treated as equal, so a
hard line break in the source matches a space in OLD. An edit applies only when its OLD occurs
exactly once in the text as it stands after the edits before it in the same file. An edit whose
NEW changes the balance of braces or of \\begin/\\end is refused unless its EDIT line carries the
tag `allow-unbalanced`. Refused, missing and ambiguous edits are reported and skipped; the rest
are written atomically. Exit status is 0 when every edit applied, 1 otherwise.
"""
import argparse
import fcntl
import hashlib
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, 'Elements_of_Bayes_INTERIOR.tex')


def parse(path):
    edits, cur, block = [], None, None
    with open(path, encoding='utf-8') as fh:
        lines = fh.read().split('\n')
    for n, line in enumerate(lines, 1):
        if line.startswith('@@@ EDIT'):
            if cur is not None:
                sys.exit(f'{path}:{n}: EDIT before END of {cur["id"]}')
            head = line[len('@@@ EDIT'):].strip()
            ident, _, tags = head.partition('|')
            cur = {'id': ident.strip() or f'line{n}', 'tags': tags.strip(), 'why': '',
                   'old': [], 'new': [], 'line': n}
            block = None
        elif cur is None:
            continue
        elif line.startswith('@@@ WHY'):
            cur['why'] = line[len('@@@ WHY'):].strip()
        elif line.strip() == '@@@ OLD':
            block = 'old'
        elif line.strip() == '@@@ NEW':
            block = 'new'
        elif line.strip() == '@@@ END':
            cur['old'] = '\n'.join(cur['old'])
            cur['new'] = '\n'.join(cur['new'])
            if not cur['old'].strip():
                sys.exit(f'{path}:{cur["line"]}: edit {cur["id"]} has an empty OLD block')
            edits.append(cur)
            cur, block = None, None
        elif block:
            cur[block].append(line)
    if cur is not None:
        sys.exit(f'{path}: edit {cur["id"]} has no END')
    return edits


def locate(text, old):
    """Return (start, end) of the single match of old, or a reason string."""
    count = text.count(old)
    if count == 1:
        start = text.index(old)
        return start, start + len(old)
    if count > 1:
        return f'ambiguous ({count} verbatim matches)'
    tokens = old.split()
    pattern = r'\s+'.join(re.escape(t) for t in tokens)
    found = [m.span() for m in re.finditer(pattern, text)]
    if len(found) == 1:
        return found[0]
    if found:
        return f'ambiguous ({len(found)} whitespace-insensitive matches)'
    return 'missing'


def balance(s):
    s = re.sub(r'\\[{}]', '', s)
    s = re.sub(r'(?<!\\)%.*', '', s)
    return (s.count('{') - s.count('}'),
            len(re.findall(r'\\begin\{', s)) - len(re.findall(r'\\end\{', s)))


def run(edits, text):
    report = []
    for e in edits:
        if 'allow-unbalanced' not in e['tags'] and balance(e['old']) != balance(e['new']):
            report.append((e, 'refused (NEW changes brace or begin/end balance)'))
            continue
        where = locate(text, e['old'])
        if isinstance(where, str):
            report.append((e, where))
            continue
        start, end = where
        text = text[:start] + e['new'] + text[end:]
        report.append((e, 'applied'))
    return text, report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mode', choices=['check', 'apply'])
    ap.add_argument('edits')
    ap.add_argument('--source', default=SOURCE)
    args = ap.parse_args()
    edits = parse(args.edits)
    source = os.path.abspath(args.source)
    key = hashlib.sha1(source.encode()).hexdigest()[:12]
    with open(os.path.join(tempfile.gettempdir(), f'edits-{key}.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(source, encoding='utf-8') as fh:
            text = fh.read()
        new_text, report = run(edits, text)
        applied = sum(r == 'applied' for _, r in report)
        if args.mode == 'apply' and applied:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(source), prefix='.edits-')
            with os.fdopen(fd, 'w', encoding='utf-8') as out:
                out.write(new_text)
            os.chmod(tmp, os.stat(source).st_mode)
            os.replace(tmp, source)
    verb = 'applied' if args.mode == 'apply' else 'would apply'
    print(f'{verb} {applied} of {len(edits)} edits to {os.path.relpath(source)}')
    for e, r in report:
        if r != 'applied':
            snippet = ' '.join(e['old'].split())[:90]
            print(f'  {e["id"]}: {r}: {snippet}')
    for e, r in report:
        if r == 'applied' and ('\u2014' in e['new'] or re.search(r'(?<!-)---(?!-)', e['new'])):
            print(f'  {e["id"]}: warning: NEW contains an em dash')
    sys.exit(0 if applied == len(edits) else 1)


if __name__ == '__main__':
    main()
