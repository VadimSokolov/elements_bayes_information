import numpy as np, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
exec(open(HERE/"roulette.py").read().split("# ---------- roulette")[0])   # universal()

T, pay, pfav = 10000, 36.0, 1/30
prest = (1-2*pfav)/36
rng = np.random.default_rng(7)
spins = rng.choice(38, size=T, p=np.array([pfav,pfav]+[prest]*36))

def waterfill(p, o=36.0):
    """Kelly with a cash reserve, equal odds o on every outcome."""
    idx = np.argsort(-p); b = np.zeros_like(p); best=None
    for k in range(1, len(p)+1):
        A = idx[:k]; P = p[A].sum(); Q = k/o
        if Q >= 1: break
        b0 = (1-P)/(1-Q)
        if p[A[-1]]*o > b0: best=(A.copy(), b0)
        else: break
    if best is None: return np.zeros_like(p), 1.0
    A, b0 = best
    b[A] = p[A] - b0/o
    return b, b0

# truth and the Kelly rule with known bias
ptrue = np.array([pfav,pfav]+[prest]*36)
btrue, b0true = waterfill(ptrue)
print("true stake per favoured pocket", btrue[0], "reserve", b0true)

# Kurtz rule: Dirichlet(1/2) posterior mean of p, then water filling
alpha = 0.5; counts = np.zeros(38); logS = 0.0; path=[]; rows={}
nA=[]; tot=[]
for t in range(1, T+1):
    pbar = (counts + alpha)/(t-1 + 38*alpha)
    b, b0 = waterfill(pbar)
    j = spins[t-1]
    logS += np.log(b0 + pay*b[j])
    counts[j] += 1
    nA.append(int((b>0).sum())); tot.append(float(b.sum()))
    if t in (100,500,1000,3000,5000,10000):
        rows[str(t)] = dict(active=int((b>0).sum()), total_stake=round(float(b.sum()),5),
                            stake_fav=round(float(b[0]+b[1]),5), logS=round(float(logS),3))
print(json.dumps(rows, indent=1))
print("Kurtz rule terminal log wealth", logS, " wealth", np.exp(logS))
json.dump(dict(rows=rows, logS=float(logS), btrue=float(btrue[0])), open(HERE/"kurtz.json","w"), indent=1)

# threshold check: Dirichlet(1/2) posterior mean clears 1/36 when n_j > (n+1)/36
print("threshold identity check:", [(n, (n+1)/36) for n in (100,1000,10000)])

# ---- sweep the prior concentration ----
def run(alpha):
    counts = np.zeros(38); logS = 0.0; snap={}
    for t in range(1, T+1):
        pbar = (counts + alpha)/(t-1 + 38*alpha)
        b, b0 = waterfill(pbar)
        j = spins[t-1]; logS += np.log(b0 + pay*b[j]); counts[j] += 1
        if t in (1000, 10000):
            snap[t] = (int((b>0).sum()), float(b.sum()), float(b[0]+b[1]))
    return logS, snap
out={}
for a in [0.5, 10, 50, 200, 882, 2000]:
    lS, sn = run(a)
    out[a] = dict(logS=round(float(lS),3), margin=round(a/18,2),
                  active1000=sn[1000][0], stake1000=round(sn[1000][1],5),
                  active10000=sn[10000][0], stake10000=round(sn[10000][1],5),
                  fav10000=round(sn[10000][2],5))
    print(a, out[a])
json.dump(out, open(HERE/"kurtz_sweep.json","w"), indent=1)
