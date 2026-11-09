import numpy as np, pathlib
exec(open(pathlib.Path(__file__).resolve().parent/'make3.py').read().split('runs={}')[0].replace('from style import *','').replace('import matplotlib.pyplot as plt',''))
q=cell*(B[:,0]*B[:,1]*B[:,2])**-0.5; q/=q.sum(); bh,ls=universal(X,B,q)
lam=(bh[:,1]+bh[:,2])/(bstar[1]+bstar[2])
print('half logS',ls[-1],'logS100',ls[99],{t:round(float(lam[t-1]),2) for t in [1,100,500,1000,3000,5000,10000]})
counts=np.zeros((T,38)); counts[np.arange(T),spins]=1; counts=counts.cumsum(0); tt=np.arange(1,T+1)
thr=tt/36+0.49*np.sqrt(tt); fire=counts.max(1)>thr
ff=np.where(counts[:,:2].max(1)>thr)[0]; fo=np.where(counts[:,2:].max(1)>thr)[0]
print('fav first fire',ff[0]+1,'other pocket fires at n>=50:',[int(x+1) for x in fo if x>=49][:20], 'count',len(fo[fo>=49]))
