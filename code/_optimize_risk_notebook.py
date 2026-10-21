import ast
import json
from pathlib import Path

path = Path('code/02_risk_matching_optimizer.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

# Update notebook narrative for sampled adaptive grids and streamed processing.
nb['cells'][0]['source'] = [
    "# Stage 2 — Adaptive Risk Matching & Pricing Optimizer\n",
    "\n",
    "Estimate local default risk and rival price pressure, then select rates by expected profit across the three borrower-preference types. This implementation uses streamed chunks, scalar precomputed local risk for the 10% gate, and per-applicant sampled rival-rate bounds.\n",
    "\n",
    "### Performance choices\n",
    "- Precompute the scalar `local_risk` vector (one float per applicant); do not cache neighbor lists across passes.\n",
    "- Process 1,000 applicants at a time and discard each chunk's neighbor arrays immediately after rate optimization.\n",
    "- For each applicant's candidate-rate grid, randomly sample about √(number of local complete-quote matches) rival pairs. Use sampled min/max plus/minus 1 percentage point as the grid bounds.\n",
    "- `H_GRID`, `WEIGHT_CUTOFF`, and `N_RATES` remain tunable. Sampling and chunk processing do not change those settings.\n",
]

# Replace function cell with efficient sampling helper and deterministic scalar risk vector helper.
nb['cells'][5]['source'] = '''def build_tree(reference_pd, h, cutoff=WEIGHT_CUTOFF):
    ref = np.asarray(reference_pd, dtype=float)
    tree = cKDTree(ref.reshape(-1, 1))
    radius = h * math.sqrt(2 * math.log(1 / cutoff))
    return ref, tree, radius

def kernel_neighbors(query_pd, ref_pd, tree, radius, h):
    """Build neighbors for one query chunk only; caller should discard after processing."""
    q = np.asarray(query_pd, dtype=float)
    groups = tree.query_ball_point(q.reshape(-1, 1), r=radius)
    result = []
    for qi, inds in zip(q, groups):
        inds = np.asarray(inds, dtype=np.int32)
        w = np.exp(-0.5 * ((qi - ref_pd[inds]) / h) ** 2)
        total = w.sum()
        if total <= 0 or not np.isfinite(total):
            inds = np.array([int(np.argmin(np.abs(ref_pd - qi)))], dtype=np.int32)
            w = np.ones(1); total = 1.0
        result.append((inds, w / total))
    return result

def get_adaptive_rate_grid(rival1, rival2, n_points=N_RATES, buffer=0.01,
                           rng=None, sample_count=None):
    """Bound this applicant's grid using a random √(matches)-sized rival-rate sample."""
    c1 = np.asarray(rival1, dtype=float)
    c2 = np.asarray(rival2, dtype=float)
    valid = np.isfinite(c1) & np.isfinite(c2)
    c1, c2 = c1[valid], c2[valid]
    n_matches = len(c1)
    if n_matches == 0:
        return np.linspace(0.03, 0.25, n_points)
    if rng is None:
        rng = np.random.default_rng(42)
    n_sample = int(np.ceil(np.sqrt(n_matches))) if sample_count is None else int(sample_count)
    n_sample = max(1, min(n_matches, n_sample))
    sample_idx = rng.choice(n_matches, size=n_sample, replace=False)
    sampled_rates = np.concatenate([c1[sample_idx], c2[sample_idx]])
    r_min = max(0.005, float(sampled_rates.min()) - buffer)
    r_max = min(1.0, float(sampled_rates.max()) + buffer)
    if r_max <= r_min:
        r_max = min(1.0, r_min + 0.02)
    return np.linspace(r_min, r_max, n_points)

def scenario_pwin_exact(rate, rival1, rival2, delta=0.0):
    """P(Win) for one specific rival pair under equal borrower preference types."""
    c1 = float(rival1) * (1.0 + delta)
    c2 = float(rival2) * (1.0 + delta)
    r = np.asarray(rate, dtype=float)
    win1 = (r < c1 - TASTE_DISCOUNT) & (r < c2)
    win2 = (r < c1) & (r < c2 - TASTE_DISCOUNT)
    win3 = (r - TASTE_DISCOUNT < c1) & (r - TASTE_DISCOUNT < c2)
    return (win1.astype(float) + win2.astype(float) + win3.astype(float)) / 3.0

def expected_profit(rate_grid, p_local, p_win):
    """P(Win) * €10,000 * [(1 - p_local) * rate - p_local]."""
    r = np.asarray(rate_grid, dtype=float)
    return np.asarray(p_win) * PRINCIPAL * ((1.0 - p_local) * r - p_local)

def p_win_by_rate(rate_grid, rival1, rival2, weights, delta=0.0):
    """Mean P(Win) over preference types for kernel-weighted rival observations."""
    c1 = np.asarray(rival1, dtype=float) * (1.0 + delta)
    c2 = np.asarray(rival2, dtype=float) * (1.0 + delta)
    r = np.asarray(rate_grid, dtype=float)[:, None]
    win1 = ((r < c1[None, :] - TASTE_DISCOUNT) & (r < c2[None, :])).astype(float)
    win2 = ((r < c1[None, :]) & (r < c2[None, :] - TASTE_DISCOUNT)).astype(float)
    win3 = ((r - TASTE_DISCOUNT < c1[None, :]) & (r - TASTE_DISCOUNT < c2[None, :])).astype(float)
    w = np.asarray(weights, dtype=float); w = w / w.sum()
    return ((win1 @ w) + (win2 @ w) + (win3 @ w)) / 3.0
'''.splitlines(keepends=True)

# Replace backtest cell: precompute scalar risk vector, then one streamed pass with per-chunk neighbor arrays.
nb['cells'][9]['source'] = '''hist_pd = hist.p_hist.to_numpy(float)
hist_default = hist.default.to_numpy(int)
hist_c1 = hist.competing1.to_numpy(float)
hist_c2 = hist.competing2.to_numpy(float)
complete_idx = np.flatnonzero(comp_mask)
bt = hist.iloc[complete_idx].reset_index(drop=True)
bt_pd = hist_pd[complete_idx]
bt_default = hist_default[complete_idx]
bt_c1 = hist_c1[complete_idx]
bt_c2 = hist_c2[complete_idx]

def precompute_local_risk(query_pd, reference_pd, defaults, h, query_global_indices=None,
                          show_progress=True, label='Local risk'):
    """Return one scalar risk estimate per query; neighbor arrays live for one chunk only."""
    ref, tree, radius = build_tree(reference_pd, h)
    q = np.asarray(query_pd, dtype=float)
    local_risk = np.full(len(q), np.nan, dtype=float)
    chunks = range(0, len(q), CHUNK_SIZE)
    iterator = tqdm(chunks, total=math.ceil(len(q) / CHUNK_SIZE), desc=label, unit='1%') if show_progress else chunks
    for start in iterator:
        stop = min(start + CHUNK_SIZE, len(q))
        matches = kernel_neighbors(q[start:stop], ref, tree, radius, h)
        for offset, (inds, w) in enumerate(matches):
            i = start + offset
            if query_global_indices is not None:
                keep = inds != query_global_indices[i]
                inds, w = inds[keep], w[keep]
            if len(inds):
                w = w / w.sum()
                local_risk[i] = float(w @ defaults[inds])
        del matches
    return local_risk

def run_backtest(h, delta=0.0, show_progress=True):
    # Scalar array only: no cached_matches or retained neighbor arrays.
    local = precompute_local_risk(bt_pd, hist_pd, hist_default, h,
                                  query_global_indices=complete_idx,
                                  show_progress=show_progress,
                                  label=f'Backtest risk h={h:.3f}')
    valid_risk = local[np.isfinite(local)]
    risk_cutoff = float(np.quantile(valid_risk, RISK_CUT_PERCENTILE))
    bids = np.full(len(bt), np.nan)
    maxev = np.full(len(bt), np.nan)
    pwin_best = np.full(len(bt), np.nan)
    rng = np.random.default_rng(10_000 + int(round(h * 1_000_000)) + int(round(delta * 10_000)))
    ref, tree, radius = build_tree(hist_pd, h)
    chunks = range(0, len(bt), CHUNK_SIZE)
    iterator = tqdm(chunks, total=math.ceil(len(bt) / CHUNK_SIZE), desc=f'Backtest optimize h={h:.3f}', unit='1%') if show_progress else chunks
    for start in iterator:
        stop = min(start + CHUNK_SIZE, len(bt))
        matches = kernel_neighbors(bt_pd[start:stop], ref, tree, radius, h)
        for offset, (inds, w) in enumerate(matches):
            row = start + offset
            if not np.isfinite(local[row]) or local[row] > risk_cutoff:
                continue
            keep = inds != complete_idx[row]
            inds, w = inds[keep], w[keep]
            if not len(inds):
                continue
            w = w / w.sum()
            cm = comp_mask[inds]
            cinds = inds[cm]
            if not len(cinds):
                continue
            rates = get_adaptive_rate_grid(hist_c1[cinds], hist_c2[cinds], n_points=N_RATES, rng=rng)
            pw = scenario_pwin_exact(rates, bt_c1[row], bt_c2[row], delta=delta)
            ev = expected_profit(rates, local[row], pw)
            j = int(np.argmax(ev))
            maxev[row] = ev[j]
            pwin_best[row] = pw[j]
            if ev[j] >= DEFAULT_PROFIT_FLOOR:
                bids[row] = rates[j]
        del matches
    c1 = bt_c1 * (1.0 + delta); c2 = bt_c2 * (1.0 + delta); typ = bt.borrowertype.to_numpy(int)
    own_effective = bids.copy(); own_effective[typ == 3] -= TASTE_DISCOUNT
    rival1_effective = c1.copy(); rival1_effective[typ == 1] -= TASTE_DISCOUNT
    rival2_effective = c2.copy(); rival2_effective[typ == 2] -= TASTE_DISCOUNT
    won = np.isfinite(bids) & (own_effective < rival1_effective) & (own_effective < rival2_effective)
    realized = np.where(won, PRINCIPAL * ((1.0 - bt_default) * bids - bt_default), 0.0)
    d_rate = float(bt_default[won].mean()) if won.any() else np.nan
    return {'h': h, 'delta': delta, 'risk_cutoff': risk_cutoff, 'n_quote_complete': len(bt),
            'n_bids': int(np.isfinite(bids).sum()), 'n_won': int(won.sum()),
            'win_rate_among_complete': float(won.mean()), 'default_rate_won': d_rate,
            'total_profit_eur': float(realized.sum()), 'mean_rate_won': float(bids[won].mean()) if won.any() else np.nan,
            'bids': bids, 'local_risk': local, 'expected_profit': maxev, 'p_win_at_optimum': pwin_best, 'won': won}

backtests = []
for h in H_GRID:
    result = run_backtest(h)
    backtests.append(result)
    print({k: v for k, v in result.items() if k not in {'bids', 'local_risk', 'expected_profit', 'p_win_at_optimum', 'won'}})
backtest_summary = pd.DataFrame([{k: v for k, v in r.items() if k not in {'bids', 'local_risk', 'expected_profit', 'p_win_at_optimum', 'won'}} for r in backtests])
display(backtest_summary)
'''.splitlines(keepends=True)

# Replace Batch 3 cell with scalar risk precompute followed by one streamed optimization pass.
nb['cells'][13]['source'] = '''def price_batch3(query_pd, h, show_progress=True):
    q = np.asarray(query_pd, dtype=float)
    n = len(q)
    # One scalar per applicant; used for the risk gate, no cached matches.
    local_risk = precompute_local_risk(q, hist_pd, hist_default, h,
                                       show_progress=show_progress,
                                       label='Batch 3 local risk')
    risk_cutoff = float(np.quantile(local_risk[np.isfinite(local_risk)], RISK_CUT_PERCENTILE))
    print(f'Batch 3 90th percentile local risk cutoff: {risk_cutoff:.2%}')
    out_rate = np.full(n, np.nan)
    out_ev = np.full(n, np.nan)
    out_pw = np.full(n, np.nan)
    rng = np.random.default_rng(42)
    ref, tree, radius = build_tree(hist_pd, h)
    chunks = range(0, n, CHUNK_SIZE)
    iterator = tqdm(chunks, total=math.ceil(n / CHUNK_SIZE), desc='Batch 3 streamed pricing', unit='1%') if show_progress else chunks
    # Single optimization pass: neighbor arrays are dropped at the end of each chunk.
    for start in iterator:
        stop = min(start + CHUNK_SIZE, n)
        matches = kernel_neighbors(q[start:stop], ref, tree, radius, h)
        for offset, (inds, w) in enumerate(matches):
            i = start + offset
            if not np.isfinite(local_risk[i]) or local_risk[i] > risk_cutoff:
                continue
            cm = comp_mask[inds]
            cinds = inds[cm]
            if not len(cinds):
                continue
            rates = get_adaptive_rate_grid(hist_c1[cinds], hist_c2[cinds], n_points=N_RATES, rng=rng)
            cw = w[cm]
            cw = cw / cw.sum()
            pw = p_win_by_rate(rates, hist_c1[cinds], hist_c2[cinds], cw)
            ev = expected_profit(rates, local_risk[i], pw)
            best = int(np.argmax(ev))
            out_ev[i] = ev[best]
            out_pw[i] = pw[best]
            if ev[best] >= DEFAULT_PROFIT_FLOOR:
                out_rate[i] = rates[best]
        del matches
    result = pd.DataFrame({'rate': out_rate, 'p_local': local_risk,
                           'max_expected_profit': out_ev, 'p_win_at_optimum': out_pw})
    return result, risk_cutoff

batch3_result, b3_risk_cutoff = price_batch3(target.p_target.to_numpy(), SELECTED_H, show_progress=True)
final_bids = pd.DataFrame({
    'id': target.id.to_numpy(),
    'rate': batch3_result.rate.to_numpy(),
    'p_default': target.p_target.to_numpy(),
    'p_local': batch3_result.p_local.to_numpy(),
    'max_expected_profit': batch3_result.max_expected_profit.to_numpy(),
    'p_win_at_optimum': batch3_result.p_win_at_optimum.to_numpy()
})
assert len(final_bids) == 100_000 and final_bids.id.is_unique
n_risk_reject = int((final_bids.p_local > b3_risk_cutoff).sum())
n_profit_reject = int((final_bids.rate.isna() & (final_bids.p_local <= b3_risk_cutoff)).sum())
print(f'Final bids issued: {final_bids.rate.notna().sum():,} | Risk rejects: {n_risk_reject:,} | Profit/market rejects: {n_profit_reject:,}')
print(f'Mean issued rate: {final_bids.rate.mean():.2%} | Mean expected profit among bids: €{final_bids.loc[final_bids.rate.notna(), "max_expected_profit"].mean():,.2f}')
display(final_bids.head())
'''.splitlines(keepends=True)

# Revise narrative to reflect actual one-scalar precomputation and streamed matching.
nb['cells'][12]['source'] = [
    "## 6. Optimize Batch 3 Rates with Adaptive Pricing & 10% Risk Cut\n",
    "\n",
    "1. Precompute only a scalar `local_risk` per applicant in chunks; discard match arrays after each chunk.\n",
    "2. Compute the 90th-percentile risk threshold from `local_risk`, rejecting the top 10%.\n",
    "3. In a single streamed pricing pass, build sampled applicant-specific rate grids and immediately optimize, without caching matches.\n",
    "4. Apply the existing configured profit floor and export diagnostics.\n",
]

# Save and validate code cells.
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']), filename=f'{path}:cell{i}')
print('Notebook optimized and all code cells parse.')
