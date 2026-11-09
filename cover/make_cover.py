# KDP paperback wraparound cover, 7 x 10 in trim, white paper.
# Usage: python3 cover/make_cover.py [--pages N] [--art PDF_OR_IMAGE]
# Page count defaults to the built interior PDF (build/Elements_of_Bayes_INTERIOR.pdf).
# Output: cover/KDP_cover_wrap.pdf. The Starry Night art (Van Gogh, public domain) is not
# stored as cover/front_art.jpg (default); --art overrides it with an image or a PDF.
# Without art the front falls back to a Polya tree drawn to twelve levels.
import argparse, os, re, subprocess, tempfile, glob
import numpy as np, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--pages', type=int)
ap.add_argument('--art', default=os.path.join(HERE, 'front_art.jpg'))
ap.add_argument('--out', default=os.path.join(HERE, 'KDP_cover_wrap.pdf'))
a = ap.parse_args()

def page_count():
    pdf = os.path.join(ROOT, 'build', 'Elements_of_Bayes_INTERIOR.pdf')
    out = subprocess.run(['pdfinfo', pdf], capture_output=True, text=True, check=True).stdout
    return int(re.search(r'Pages:\s+(\d+)', out).group(1))

PAGES = a.pages or page_count()
TRIM_W, TRIM_H, BLEED = 7.0, 10.0, 0.125
SPINE = PAGES * 0.002252                      # KDP white paper
W = 2 * TRIM_W + SPINE + 2 * BLEED; H = TRIM_H + 2 * BLEED
BLUE = '#1e4a86'; NAVY_T = '#14315d'; NAVY_B = '#081731'; GOLD = '#e0b84a'

def lmfont(fname, family):
    p = subprocess.run(['kpsewhich', fname], capture_output=True, text=True).stdout.strip()
    return fm.FontProperties(fname=p) if p else fm.FontProperties(family=family)
CAPS = lmfont('lmromancaps10-regular.otf', 'Latin Modern Roman Caps')
ROMAN = lmfont('lmroman10-regular.otf', 'Latin Modern Roman')
ITAL = lmfont('lmromanslant10-regular.otf', 'Latin Modern Roman Slanted')
mpl.rcParams.update({'pdf.fonttype': 42, 'mathtext.fontset': 'cm'})
def T(x, y, s, fam, size, color='white', weight='normal', style='normal', **k):
    return ax.text(x, y, s, fontproperties=fam.copy(), fontsize=size, color=color, **k)

fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
ax.add_patch(Rectangle((0, 0), W, H, color=BLUE, zorder=0))
back_x0 = BLEED; spine_x0 = BLEED + TRIM_W; front_x0 = spine_x0 + SPINE

def vgrad(x0, x1, y0, y1, c0, c1):
    g = np.linspace(0, 1, 256)[:, None]
    c0 = np.array(mpl.colors.to_rgb(c0)); c1 = np.array(mpl.colors.to_rgb(c1))
    img = (c0 * (1 - g) + c1 * g)[:, None, :]          # row 0 = top
    ax.imshow(img, extent=(x0, x1, y0, y1), aspect='auto', zorder=1, interpolation='bilinear')
    ax.set_xlim(0, W); ax.set_ylim(0, H)

# spine: dark gradient, full height including bleed
vgrad(spine_x0, front_x0, 0, H, NAVY_T, NAVY_B)

# ---- front ----
cx = front_x0 + TRIM_W / 2
art_top = H - 3.15; art_w = TRIM_W + BLEED       # runs off the right bleed edge
def get_art():
    if not os.path.exists(a.art): return None
    if not a.art.lower().endswith('.pdf'): return plt.imread(a.art)
    d = tempfile.mkdtemp()
    subprocess.run(['pdfimages', '-j', a.art, os.path.join(d, 'p')], check=True)
    f = sorted(glob.glob(os.path.join(d, 'p-*.jpg')))
    return plt.imread(f[0]) if f else None
art = get_art()
if art is not None:
    art_h = art_w * art.shape[0] / art.shape[1]
    ax.imshow(art, extent=(front_x0, front_x0 + art_w, art_top - art_h, art_top), aspect='auto', zorder=2)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    vgrad(front_x0, W, 0, art_top - art_h, '#10285a', NAVY_B)   # darkening band under the art
else:
    def polya_tree(seed=11, DEPTH=12):
        rng = np.random.default_rng(seed); segs = []; deps = []; mas = []
        stack = [(0.0, 0.0, np.pi / 2, 1.0, 0, 1.0)]
        while stack:
            x, y, ang, length, depth, mass = stack.pop()
            if depth > DEPTH or length < 0.004: continue
            v = rng.beta(5.0 + 1.5 * depth, 5.0 + 1.5 * depth)
            x2 = x + length * np.cos(ang); y2 = y + length * np.sin(ang)
            segs.append([(x, y), (x2, y2)]); deps.append(depth); mas.append(mass)
            sp = 0.62 - 0.018 * depth
            stack.append((x2, y2, ang + sp * (1 - v) * 1.9, length * (0.66 + 0.13 * v), depth + 1, mass * v))
            stack.append((x2, y2, ang - sp * v * 1.9, length * (0.66 + 0.13 * (1 - v)), depth + 1, mass * (1 - v)))
        return np.array(segs), np.array(deps), np.array(mas)
    segs, deps, mas = polya_tree()
    cols = ['#37587f', '#44749a', '#5194a4', '#5fa39b', '#79b090', '#9bba73', '#bfb23f', '#d4a11e', '#e4c85a']
    PW, PH = TRIM_W - 1.0, 4.3
    axf = fig.add_axes([(front_x0 + 0.5) / W, (H - 3.15 - PH) / H, PW / W, PH / H], zorder=2); axf.axis('off')
    axf.add_collection(LineCollection(segs, colors=[cols[min(d, 8)] for d in deps],
        linewidths=[max(0.3, 3.2 * m ** 0.4) for m in mas], capstyle='round', alpha=0.93))
    xs = segs[:, :, 0].ravel(); ys = segs[:, :, 1].ravel()
    yr = (ys.max() - ys.min()) * 1.06; xr = yr * (PW / PH)
    axf.set_xlim((xs.min() + xs.max()) / 2 - xr / 2, (xs.min() + xs.max()) / 2 + xr / 2)
    axf.set_ylim((ys.min() + ys.max()) / 2 - yr / 2, (ys.min() + ys.max()) / 2 + yr / 2)

T(cx, H - 1.17, 'Elements of Bayes and', CAPS, 36, ha='center', va='center', zorder=5)
T(cx, H - 1.95, 'Information Theory', CAPS, 36, ha='center', va='center', zorder=5)
T(cx, H - 2.72, 'A Predictive Theory of Inference', ITAL, 24, ha='center', va='center', zorder=5)
T(cx, 0.95, 'Nick Polson and Vadim Sokolov', ROMAN, 28, ha='center', va='center', zorder=5)

# ---- spine (text reads top to bottom) ----
sx = spine_x0 + SPINE / 2
tsz = min(19, (SPINE - 0.125) * 72 * 0.62)
T(sx, H - 0.75, 'Elements of Bayes and Information Theory', CAPS, tsz,
  rotation=270, ha='center', va='top', zorder=5)
T(sx, 0.75 + 2.3, 'Polson • Sokolov', ROMAN, tsz * 0.72,
  rotation=270, ha='center', va='top', zorder=5)

# ---- back ----
bx = back_x0 + 0.85; bw_text = TRIM_W - 0.85 - 0.6
T(back_x0 + TRIM_W / 2, H - 1.05, 'Probability does not exist.', ITAL, 15, style='italic', ha='center', va='center')
T(back_x0 + TRIM_W / 2, H - 1.62, 'Bruno de Finetti', ROMAN, 10, color=GOLD, ha='center', va='center')
paras = [
 "Bayesian inference is a predictive act. This book builds it from one idea, prediction, and shows that de Finetti’s representation, Sanov’s theorem and Bayes’ rule are three views of a single calculation.",
 "Fourteen chapters carry the argument from exchangeability, types and large deviations, through coding, shrinkage, maximum entropy, e-values and testing, to posterior asymptotics, scalable computation and deep learning, growth optimal investment, reinforcement learning, causal machine learning, medical decision making and federated learning.",
 "Unlike treatments that present Bayes as a recipe for posteriors, this book treats every inference as a decision on one moderate deviation scale. Readers learn why a Bayesian test threshold grows with the sample size, how an e-value stays valid under continuous monitoring, why a universal portfolio is a posterior mean, and how a calibrated prior can change a cancer treatment decision."]
import textwrap
y = H - 2.15
for p in paras:
    txt = '\n'.join(textwrap.wrap(p, 80))
    T(bx, y, txt, ROMAN, 11, color='#f0f3f8', va='top', linespacing=1.45)
    y -= 0.185 * (txt.count('\n') + 1) + 0.22
for y0, name, rest in [(2.95, 'Nick Polson', ' is the Robert Law Jr. Professor of Econometrics and Statistics at the Booth School of Business, University of Chicago.'),
                       (2.25, 'Vadim Sokolov', ' is an Associate Professor in Systems Engineering and Operations Research at George Mason University.')]:
    lines = textwrap.wrap(name + rest, 62)
    first = lines[0][len(name):]
    T(bx, y0, name, ROMAN, 9, va='top')
    r = fig.canvas.get_renderer()
    w = ax.texts[-1].get_window_extent(r).width / fig.dpi
    T(bx + w, y0, first, ROMAN, 9, va='top')
    T(bx, y0 - 0.17, '\n'.join(lines[1:]), ROMAN, 9, va='top', linespacing=1.5)
T(bx, 1.65, 'Cover: Vincent van Gogh, Starry Night Over the Rhône, 1888,\nMusée d’Orsay, Paris. Public domain.',
  ITAL, 8, color=GOLD, style='italic', va='top', linespacing=1.6)
# KDP barcode clear area, 2 x 1.2 in, lower right of the back cover (kept empty, no drawing)

fig.savefig(a.out)
print('cover %.4f x %.4f in, spine %.4f in for %d pages' % (W, H, SPINE, PAGES))
