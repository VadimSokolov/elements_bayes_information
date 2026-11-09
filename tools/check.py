#!/usr/bin/env python3
"""Summarise the last build and run the house-style audits; exit 1 on any hard failure.

    python3 tools/check.py                # one-screen summary (`make` runs this)
    python3 tools/check.py --overfull 1   # also list every overfull box wider than 1 pt
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BOOK = 'Elements_of_Bayes_INTERIOR'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--overfull', type=float, metavar='PT', help='list overfull boxes wider than PT')
    ap.add_argument('--build', default='build', help='build directory holding the log (default build/)')
    args = ap.parse_args()
    log = (ROOT / args.build / f'{BOOK}.log').read_text(errors='replace')
    tex = (ROOT / f'{BOOK}.tex').read_text(encoding='utf-8').split('\n')
    prose = [line for line in tex if not line.lstrip().startswith('%')]

    hard = {
        'errors': len(re.findall(r'^! ', log, re.M)),
        'undefined references': len(re.findall(r'LaTeX Warning: (?:Reference|Citation) .* undefined', log)),
        'multiply defined labels': len(re.findall(r'LaTeX Warning: Label .* multiply defined', log)),
        'missing files': len(re.findall(r"File `[^']*' not found|Unable to load picture", log)),
        'prose em dashes': sum(bool(re.search(r'(?<!-)---(?!-)', line)) for line in prose),
        'unicode dashes': sum(('—' in line or '–' in line) for line in tex),
        'boxed equations': sum(line.count('\\boxed') for line in prose),
    }
    boxes = [(float(w), where.strip())
             for w, where in re.findall(r'^Overfull \\hbox \(([\d.]+)pt too wide\)(.*)$', log, re.M)]
    pages = re.search(r'Output written on \S+ \((\d+) pages', log)

    print(f'{pages.group(1) if pages else "no PDF"} pages | '
          + ' | '.join(f'{k} {v}' for k, v in hard.items() if k in list(hard)[:4]))
    if boxes:
        worst = max(boxes)
        print(f'overfull boxes {len(boxes)}: {sum(w > 1 for w, _ in boxes)} over 1 pt, '
              f'{sum(w > 5 for w, _ in boxes)} over 5 pt, worst {worst[0]:.1f} pt ({worst[1]})')
    else:
        print('overfull boxes 0')
    print('style: ' + ' | '.join(f'{k} {v}' for k, v in hard.items() if k in list(hard)[4:]))
    if 'Rerun to get' in log:
        print('warning: LaTeX asks for another pass')
    if args.overfull is not None:
        for w, where in sorted(boxes, key=lambda b: int((re.findall(r'\d+', b[1]) or ['0'])[0])):
            if w > args.overfull:
                print(f'  {w:6.1f} pt  {where}')
    failed = [k for k, v in hard.items() if v]
    print('FAIL: ' + ', '.join(failed) if failed else 'OK')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
