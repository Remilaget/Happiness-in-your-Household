import math
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Load data
b2 = pd.read_csv('data/processed/Batch2.csv')
p_new = pd.read_csv('data/processed/batch2_p_default_catboost.csv')
b2['p_hist'] = p_new['pd']
comp_mask = b2[['competing1', 'competing2']].notna().all(axis=1).to_numpy()

hist_pd = b2['p_hist'].to_numpy(float)
hist_default = b2['default'].to_numpy(int)
hist_c1 = b2['competing1'].to_numpy(float)
hist_c2 = b2['competing2'].to_numpy(float)

h = 0.010
cutoff = 1e-4
radius = h * math.sqrt(2 * math.log(1 / cutoff))
inv_h2 = -0.5 / (h * h)

# Test 1: CPU Streaming without memory leak
t0 = time.time()
tree = cKDTree(hist_pd.reshape(-1, 1))

# Pass 1: compute local risk only (stores only a 1D float array of 100k numbers = 800 KB!)
local_risk = np.full(len(b2), np.nan)
CHUNK_SIZE = 2500
for start in range(0, len(b2), CHUNK_SIZE):
    stop = min(start + CHUNK_SIZE, len(b2))
    groups = tree.query_ball_point(hist_pd[start:stop].reshape(-1, 1), r=radius)
    for k, inds in enumerate(groups):
        row = start + k
        inds = np.asarray(inds, dtype=np.int32)
        # Leave-one-out
        inds = inds[inds != row]
        if len(inds):
            w = np.exp(inv_h2 * (hist_pd[row] - hist_pd[inds]) ** 2)
            local_risk[row] = float(w @ hist_default[inds] / w.sum())

risk_cutoff = float(np.nanquantile(local_risk, 0.90))
t1 = time.time()
print(f'CPU Streaming Pass 1 (100k local risk + 90% cutoff): {t1 - t0:.2f}s | Cutoff: {risk_cutoff:.4f}')

# Check memory: local_risk is just 800 KB
print(f'Memory of local_risk array: {local_risk.nbytes / 1024:.1f} KB (NO 14 GB LEAK!)')

