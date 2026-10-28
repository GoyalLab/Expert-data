import sys
from pathlib import Path

import h5py, numpy as np, time, os, tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _requires import requires_env  # noqa: E402
requires_env('GPFS_OUT', what='a shared-filesystem scratch dir to benchmark against')
rng=np.random.default_rng(0)
N=60_000_000
d=rng.integers(0,100,N,dtype=np.int16); i=rng.integers(0,3000,N,dtype=np.int32)
raw=(d.nbytes+i.nbytes)/1024**2
print(f'  writing {raw:.0f} MB raw in 20 incremental resize+append steps\n')
print(f'  {"target":34s} {"comp":6s} {"sec":>7s} {"MB/s":>8s}')
for tgt,lbl in ((os.environ['GPFS_OUT'],'GPFS (project scratch)'),('/tmp','node-local SSD (/tmp)')):
    for comp,copts in ((None,{}),('gzip',{'compression':'gzip','compression_opts':1})):
        p=os.path.join(tgt,f'wb_{os.getpid()}_{comp}.h5')
        t0=time.time()
        with h5py.File(p,'w') as f:
            g=f.create_group('X')
            od=g.create_dataset('data',(0,),maxshape=(None,),dtype=np.int16,chunks=(1_000_000,),**copts)
            oi=g.create_dataset('indices',(0,),maxshape=(None,),dtype=np.int32,chunks=(1_000_000,),**copts)
            step=N//20; w=0
            for a in range(0,N,step):
                b=min(a+step,N)
                od.resize(w+(b-a),axis=0); oi.resize(w+(b-a),axis=0)
                od[w:w+(b-a)]=d[a:b]; oi[w:w+(b-a)]=i[a:b]; w+=b-a
        el=time.time()-t0; os.unlink(p)
        print(f'  {lbl:34s} {str(comp):6s} {el:7.1f} {raw/el:8.1f}')
