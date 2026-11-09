import numpy as np, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent

# ---------- multiplicative grid update (no exponentials) ----------
def universal(X, B, q):
    """X: T x n price relatives. B: N x n grid nodes. q: prior quadrature weights (sum 1)."""
    T = X.shape[0]; w = q.copy()
    bhat = np.empty((T, B.shape[1])); logrel = np.empty(T)
    for t in range(T):
        b = w @ B                       # posterior mean = portfolio
        bhat[t] = b
        r = B @ X[t]                    # wealth relatives of every node
        w = w * r
        Z = w.sum()                     # equals b . x_t exactly
        w /= Z
        logrel[t] = np.log(Z)
    return bhat, np.cumsum(logrel)

# ---------- roulette: American wheel, two favoured pockets ----------
T, pay, pfav = 10000, 36.0, 1.0/30.0
prest = (1 - 2*pfav)/36
rng = np.random.default_rng(7)
probs = np.array([pfav, pfav] + [prest]*36)
spins = rng.choice(38, size=T, p=probs)
X = np.zeros((T, 3)); X[:, 0] = 1.0      # assets: cash, number 1, number 2
X[spins == 0] = [1.0, pay, 0.0]
X[spins == 1] = [1.0, 0.0, pay]

# analytic Kelly rule with reserve (threshold / water filling)
PA, QA = 2*pfav, 2/pay
b0 = (1-PA)/(1-QA); bstar = np.array([b0, pfav-b0/pay, pfav-b0/pay])
Wstar = 2*pfav*np.log(pfav*pay) + (1-PA)*np.log(b0)

# graded grid on the 2 simplex, cubic grading toward the cash vertex
N = 300
edge = (np.arange(N+1)/N)**3
mid, wid = (edge[:-1]+edge[1:])/2, np.diff(edge)
M1, M2 = np.meshgrid(mid, mid, indexing="ij"); W1, W2 = np.meshgrid(wid, wid, indexing="ij")
keep = (M1+M2) < 1
b1, b2, cell = M1[keep], M2[keep], (W1*W2)[keep]
B = np.column_stack([1-b1-b2, b1, b2])
q = cell*np.prod(B, axis=1)**-0.5        # Dirichlet(1/2,1/2,1/2)
q /= q.sum()
print("grid nodes", len(q))

bhat, logSu = universal(X, B, q)
logSgrid = (B @ np.log(np.array([[1.,pay,0.],[1.,0.,pay],[1.,1.,1.]]).T*0+1)).sum()*0  # placeholder
n1, n2 = int((spins==0).sum()), int((spins==1).sum()); n0 = T-n1-n2
logS_all = n1*np.log(B[:,0]+pay*B[:,1]) + n2*np.log(B[:,0]+pay*B[:,2]) + n0*np.log(B[:,0])
k = int(np.argmax(logS_all))
ethier_n = None
tt = np.arange(1,T+1)
counts = np.zeros((T,38)); counts[np.arange(T), spins] = 1; counts = counts.cumsum(axis=0)
thr = tt/36 + 0.49*np.sqrt(tt)
fired = np.where(counts.max(axis=1) > thr)[0]
argfire = int(counts[fired[0]].argmax()) if len(fired) else -1
res = dict(T=T, n1=n1, n2=n2, bstar=bstar.tolist(), Wstar=Wstar, TWstar=T*Wstar,
   grid_best_b=B[k].tolist(), logS_best=float(logS_all[k]),
   logS_star_analytic=float(n1*np.log(bstar[0]+pay*bstar[1])+n2*np.log(bstar[0]+pay*bstar[2])+n0*np.log(bstar[0])),
   logS_cash=0.0, logS_univ=float(logSu[-1]), S_univ=float(np.exp(logSu[-1])),
   regret=float(logS_all[k]-logSu[-1]), bound=float(np.log(2)+np.log(T+1)),
   bhat_final=bhat[-1].tolist(), ethier_first_fire=int(fired[0]+1) if len(fired) else None,
   ethier_expected_n=float((0.49*180)**2), ethier_number=argfire)
print(json.dumps(res, indent=1))
res["bhat_at"] = {str(k): bhat[k-1].tolist() for k in [100,500,1000,3000,5000,10000]}
res["logSu_at"] = {str(k): float(logSu[k-1]) for k in [1000,5000,10000]}
tot = bhat[:,1]+bhat[:,2]
res["first_half_stake"] = int(np.argmax(tot > 0.5*2*bstar[1]))+1
json.dump(res, open(HERE/"roulette.json","w"), indent=1)

# ---------- grid convergence, Parrondo case ----------
Tp, f, p = 5000, 0.05, 0.51
r = np.random.default_rng(51); Xp = np.where(r.random((Tp,2))<p, 1+f, 1-f)
rows=[]
for Ng in [11, 26, 51, 101, 401, 1601, 4001, 16001]:
    e = np.arange(Ng+1)/Ng; m=(e[:-1]+e[1:])/2
    Bg = np.column_stack([m, 1-m]); qg = (m*(1-m))**-0.5; qg/=qg.sum()
    bh, lS = universal(Xp, Bg, qg)
    rows.append((Ng, float(lS[-1]), float(bh[-1,0])))
    print(rows[-1])
json.dump(rows, open(HERE/"gridconv.json","w"), indent=1)
