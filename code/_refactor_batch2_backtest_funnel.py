import ast, json
from pathlib import Path

path = Path('code/02_risk_matching_optimizer.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

# Preserve the user's square-root quote subsampling; add optional RNG for reproducibility/caller compatibility.
s = ''.join(nb['cells'][5]['source'])
start = s.index('def get_adaptive_rate_grid(')
end = s.index('\ndef scenario_pwin_exact', start)
s = s[:start] + '''def get_adaptive_rate_grid(rival1, rival2, n_points=10, buffer=0.01, rng=None):
    """Build a compact applicant-specific grid from sqrt-sized samples of both quote arrays."""
    r1 = np.asarray(rival1, dtype=float)
    r2 = np.asarray(rival2, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    r1 = r1[np.isfinite(r1)]; r2 = r2[np.isfinite(r2)]
    r1_sample = rng.choice(r1, size=max(1, int(np.sqrt(len(r1)))), replace=False) if len(r1) else np.array([])
    r2_sample = rng.choice(r2, size=max(1, int(np.sqrt(len(r2)))), replace=False) if len(r2) else np.array([])
    sampled = np.concatenate([r1_sample, r2_sample])
    if not len(sampled):
        return np.linspace(0.03, 0.25, n_points)
    r_min = max(0.0, float(sampled.min()) - buffer)
    r_max = min(1.0, float(sampled.max()) + buffer)
    if r_max <= r_min:
        r_max = min(1.0, r_min + 0.05)
    return np.linspace(r_min, r_max, n_points)
''' + s[end:]
nb['cells'][5]['source'] = s.splitlines(keepends=True)

nb['cells'][0]['source'] = '''# Stage 2 — Sequential Risk-Matching & Pricing Optimizer

Estimate kernel-local risk and rival price pressure, apply the requested staged funnel to the Batch 2 backtest and Batch 3 pricing, and retain all Batch 3 IDs with missing rates for rejected applicants.
'''.splitlines(keepends=True)
nb['cells'][1]['source'] = ''.join(nb['cells'][1]['source']).replace("RISK_CUT_PERCENTILE = 0.90     # Cut the 10% most likely to default\nPROFIT_CREAM_PERCENTILE = 0.005 # Cream at most 1% on expected profit\nDEFAULT_PROFIT_FLOOR = 5.0     # Baseline floor (~1% of positive EV)", "FUNNEL_RATE = 0.035           # Sequentially reject 3.5% of remaining applicants per stage\nRISK_CUT_PERCENTILE = 1.0 - FUNNEL_RATE\nPROFIT_CREAM_PERCENTILE = FUNNEL_RATE").replace("print('Configuration initialized with adaptive grid and dynamic risk cutoff.')", "print('Configuration initialized with adaptive rate grid and sequential 3.5% funnel.')").splitlines(keepends=True)

nb['cells'][8]['source'] = '''## 4. Batch 2 Backtest: Sequential local-risk and expected-profit cuts

Use the quote-complete Batch 2 cohort for a valid realized-win replay. Estimate leave-one-out local PD for this cohort, reject exactly the top 3.5% by local risk, calculate rates and expected profit only for the remaining applicants, then reject applicants below their cohort's 3.5th-percentile expected profit. Batch 2 has no independent initial global PD file in this workflow, so the backtest uses its two available stages (local risk, expected profit); Batch 3 additionally uses the initial global CatBoost-PD stage.
'''.splitlines(keepends=True)

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

def exact_top_fraction_mask(values, fraction):
    """Choose exactly floor(fraction*n) largest values; stable ordering resolves ties."""
    values = np.asarray(values, dtype=float)
    n_reject = int(np.floor(fraction * len(values)))
    mask = np.zeros(len(values), dtype=bool)
    if n_reject:
        mask[np.argsort(-values, kind='stable')[:n_reject]] = True
    return mask

def run_backtest(h, delta=0.0, show_progress=True):
    n = len(bt)
    local = np.full(n, np.nan)
    bids = np.full(n, np.nan)
    maxev = np.full(n, np.nan)
    pwin_best = np.full(n, np.nan)
    rng = np.random.default_rng(42)
    ref, tree, radius = build_tree(hist_pd, h)

    # Stage 1: Batch 2 has no separate initial PD target file; this backtest starts at local risk.
    # Compute leave-one-out local PD in chunks and discard each chunk's neighbor arrays immediately.
    chunks = range(0, n, CHUNK_SIZE)
    iterator = tqdm(chunks, total=math.ceil(n / CHUNK_SIZE), desc=f'Backtest h={h:.3f}, delta={delta:+.0%}: local PD', unit='1%') if show_progress else chunks
    for start in iterator:
        stop = min(start + CHUNK_SIZE, n)
        neighbors = kernel_neighbors(bt_pd[start:stop], ref, tree, radius, h)
        for k, (inds, weights) in enumerate(neighbors):
            row = start + k
            inds_global = inds[inds != complete_idx[row]]
            weights = weights[inds != complete_idx[row]]
            if len(inds_global):
                weights = weights / weights.sum()
                local[row] = float(weights @ hist_default[inds_global])
        del neighbors

    valid = np.isfinite(local)
    if not valid.all():
        raise ValueError(f'Local PD missing for {int((~valid).sum())} Batch 2 backtest applicants')
    risk_cutoff = float(np.quantile(local, 1.0 - FUNNEL_RATE))
    risk_reject = exact_top_fraction_mask(local, FUNNEL_RATE)
    local_survivors = ~risk_reject

    # Stage 2: Optimize expected profit/rate only for local-risk survivors.
    chunks = range(0, n, CHUNK_SIZE)
    iterator = tqdm(chunks, total=math.ceil(n / CHUNK_SIZE), desc=f'Backtest h={h:.3f}, delta={delta:+.0%}: pricing', unit='1%') if show_progress else chunks
    for start in iterator:
        stop = min(start + CHUNK_SIZE, n)
        query_rows = np.flatnonzero(local_survivors[start:stop]) + start
        if not len(query_rows):
            continue
        neighbors = kernel_neighbors(bt_pd[query_rows], ref, tree, radius, h)
        for row, (inds, weights) in zip(query_rows, neighbors):
            keep = inds != complete_idx[row]
            inds = inds[keep]
            weights = weights[keep]
            # Only quote-complete matched peers can inform the adaptive rate grid.
            cm = comp_mask[inds]
            cinds = inds[cm]
            if not len(cinds):
                continue
            rates = get_adaptive_rate_grid(hist_c1[cinds], hist_c2[cinds], n_points=N_RATES, rng=rng)
            pwin = scenario_pwin_exact(rates, bt_c1[row], bt_c2[row], delta=delta)
            ev = expected_profit(rates, local[row], pwin)
            best = int(np.argmax(ev))
            maxev[row] = ev[best]
            pwin_best[row] = pwin[best]
            bids[row] = rates[best]
        del neighbors

    priced = local_survivors & np.isfinite(maxev)
    if not priced.any():
        raise ValueError('No Stage 2 Batch 2 applicants had enough usable matched quotes to price')
    profit_cutoff = float(np.quantile(maxev[priced], FUNNEL_RATE))
    profit_reject = priced & (maxev < profit_cutoff)
    bids[~priced | risk_reject | profit_reject] = np.nan

    c1 = bt_c1 * (1.0 + delta); c2 = bt_c2 * (1.0 + delta); typ = bt.borrowertype.to_numpy(int)
    own_effective = bids.copy(); own_effective[typ == 3] -= TASTE_DISCOUNT
    rival1_effective = c1.copy(); rival1_effective[typ == 1] -= TASTE_DISCOUNT
    rival2_effective = c2.copy(); rival2_effective[typ == 2] -= TASTE_DISCOUNT
    won = np.isfinite(bids) & (own_effective < rival1_effective) & (own_effective < rival2_effective)
    realized = np.where(won, PRINCIPAL * ((1.0 - bt_default) * bids - bt_default), 0.0)
    d_rate = float(bt_default[won].mean()) if won.any() else np.nan
    return {'h': h, 'delta': delta, 'risk_cutoff': risk_cutoff, 'profit_cutoff': profit_cutoff,
            'n_quote_complete': n, 'n_stage1_rejected': int(risk_reject.sum()),
            'n_stage2_rejected': int(profit_reject.sum()), 'n_bids': int(np.isfinite(bids).sum()),
            'n_won': int(won.sum()), 'win_rate_among_complete': float(won.mean()),
            'default_rate_won': d_rate, 'total_profit_eur': float(realized.sum()),
            'mean_rate_won': float(bids[won].mean()) if won.any() else np.nan,
            'bids': bids, 'local_risk': local, 'expected_profit': maxev,
            'p_win_at_optimum': pwin_best, 'won': won}

backtests = []
for h in H_GRID:
    result = run_backtest(h)
    backtests.append(result)
    print({k: v for k, v in result.items() if k not in {'bids', 'local_risk', 'expected_profit', 'p_win_at_optimum', 'won'}})
backtest_summary = pd.DataFrame([{k: v for k, v in r.items() if k not in {'bids', 'local_risk', 'expected_profit', 'p_win_at_optimum', 'won'}} for r in backtests])
display(backtest_summary)
'''.splitlines(keepends=True)

# Make stress-test summary ignore any additional arrays only; existing exclusion sufficient.
nb['cells'][13]['source'] = ''.join(nb['cells'][13]['source']).replace('FUNNEL_RATE = 0.035\n\ndef exact_top_fraction_mask(values, fraction):\n    """Select exactly floor(fraction*n) largest values, with stable tie handling."""\n    values = np.asarray(values, dtype=float)\n    n_reject = int(np.floor(fraction * len(values)))\n    mask = np.zeros(len(values), dtype=bool)\n    if n_reject:\n        order = np.argsort(-values, kind=\'stable\')\n        mask[order[:n_reject]] = True\n    return mask\n\n', '')
s13 = ''.join(nb['cells'][13]['source'])
s13 = s13.replace('rates = get_adaptive_rate_grid(hist_c1[cinds], hist_c2[cinds], n_points=N_RATES, rng=rng)', 'rates = get_adaptive_rate_grid(hist_c1[cinds], hist_c2[cinds], n_points=N_RATES, rng=rng)')
nb['cells'][13]['source'] = s13.splitlines(keepends=True)

# Keep required exports of the backtest summaries and latest Batch 3 outputs.
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']), filename=f'{path}:cell{i}')
print('Notebook updated; all code cells parse.')
