"""Where does materialize_part actually spend its time? Phase-timed, one dataset."""
import sys, time
from pathlib import Path
sys.path.insert(0, '/path/to/projects/perturbation_prediction/ExPert/ExPert-data')
import numpy as np, h5py
from src.select import load_sidecars, scale_efficiency

ds = sys.argv[1]; arrays = sys.argv[2]; store = Path(sys.argv[3])
t = {}
t0 = time.time(); z = np.load(arrays, allow_pickle=False)
col_idx = z['col_idx']; mask = z[f'mask__{ds}']
zs = z[f'zs__{ds}'] if f'zs__{ds}' in z.files else None
cmk = z[f'colmask__{ds}'] if f'colmask__{ds}' in z.files else None
t['load recipe'] = time.time()-t0
t0 = time.time(); sc = {ds: load_sidecars(store, ds)}; t['load_sidecars'] = time.time()-t0
t0 = time.time(); eff = scale_efficiency(sc, {ds: mask}); t['scale_efficiency'] = time.time()-t0
obs = sc[ds]['obs']
is_ctrl = obs['is_ctrl'].to_numpy().astype(bool)
route = {'pert': mask & ~is_ctrl, 'ctrl': mask & is_ctrl}
if zs is not None: route['zs'] = zs
n_genes = len(col_idx)
blk = store/'cells'/f'{ds}.h5ad'
t_read = t_route = t_write = 0.0; nnz_in = nnz_out = 0
import tempfile, os
tf = tempfile.NamedTemporaryFile(suffix='.h5', delete=False); tf.close()
with h5py.File(str(blk)) as f, h5py.File(tf.name,'w') as fo:
    X=f['X']; n_obs,n_var=(int(v) for v in X.attrs['shape'])
    cm = np.full(n_var,-1,np.int32); cm[col_idx]=np.arange(n_genes,dtype=np.int32)
    if cmk is not None: cm[col_idx[~cmk]] = -1
    ip=X['indptr'][:]; data,indices=X['data'],X['indices']
    od=fo.create_dataset('d',(0,),maxshape=(None,),dtype=np.int16,chunks=(1_000_000,))
    oi=fo.create_dataset('i',(0,),maxshape=(None,),dtype=np.int32,chunks=(1_000_000,))
    touched=np.zeros(n_obs,bool)
    for r in route.values(): touched|=r
    w=0
    for s in range(0,n_obs,20_000):
        e=min(s+20_000,n_obs)
        if not touched[s:e].any(): continue
        a,b=int(ip[s]),int(ip[e]); nnz_in+=b-a
        t0=time.time(); d=np.asarray(data[a:b]); idx=np.asarray(indices[a:b]); t_read+=time.time()-t0
        t0=time.time()
        lengths=(ip[s+1:e+1]-ip[s:e]).astype(np.intp)
        mapped=cm[idx]; valid=mapped>=0
        outs=[]
        for k in route:
            sel=route[k][s:e]
            if not sel.any(): continue
            keep=valid&np.repeat(sel,lengths)
            outs.append((d[keep],mapped[keep]))
        t_route+=time.time()-t0
        t0=time.time()
        for dd,mm in outs:
            od.resize(w+dd.size,axis=0); oi.resize(w+dd.size,axis=0)
            od[w:w+dd.size]=dd; oi[w:w+dd.size]=mm; w+=dd.size
        t_write+=time.time()-t0
    nnz_out=w
os.unlink(tf.name)
tot=t_read+t_route+t_write
print(f'\n{ds}: {n_obs:,} rows, {nnz_in/1e9:.2f}B nnz read -> {nnz_out/1e9:.3f}B out')
for k,v in t.items(): print(f'  {k:18s} {v:8.1f}s')
print(f'  {"read":18s} {t_read:8.1f}s  {100*t_read/tot:5.1f}%')
print(f'  {"route":18s} {t_route:8.1f}s  {100*t_route/tot:5.1f}%')
print(f'  {"write":18s} {t_write:8.1f}s  {100*t_write/tot:5.1f}%')
print(f'  {"TOTAL loop":18s} {tot:8.1f}s')
