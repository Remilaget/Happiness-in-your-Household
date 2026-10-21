import ast, json
from pathlib import Path

path = Path('code/02_risk_matching_optimizer.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

# Correct the shared funnel rate and make percentiles consistent.
s = ''.join(nb['cells'][1]['source'])
s = s.replace('FUNNEL_RATE = 0.034', 'FUNNEL_RATE = 0.035')
nb['cells'][1]['source'] = s.splitlines(keepends=True)

# Merge Batch 2 model PDs into historical cases for Stage 1 and keep them aligned by ID.
s = ''.join(nb['cells'][3]['source'])
s = s.replace("hist = batch2.merge(batch2_pd[['id', 'pd']].rename(columns={'pd': 'p_hist'}), on='id', validate='one_to_one')", "hist = batch2.merge(batch2_pd[['id', 'pd']].rename(columns={'pd': 'p_global'}), on='id', validate='one_to_one')\nhist = hist.rename(columns={'p_global': 'p_hist'})")
# retain existing p_hist use but explicit Batch2 PD column exists from the requested file
nb['cells'][3]['source'] = s.splitlines(keepends=True)

# Replace the Batch 2 cohort setup / backtest implementation.
nb['cells'][9]['source'] = '''hist_pd = hist.p_hist.to_numpy(float)
hist_default = hist.default.to_numpy(int)
hist_c1 = hist.competing1.to_numpy(float)
hist_c2 = hist.competing2.to_numpy(float)
complete_idx = np.flatnonzero(comp_mask)
bt = hist.iloc[complete_idx].reset_index(drop=True)
bt_pd = hist_pd[complete_idx]
bt_global_pd = bt.p_hist.to_numpy(float)
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
    global_reject = exact_top_fraction_mask(bt_global_pd, FUNNEL_RATE)
    global_cutoff = float(np.quantile(bt_global_pd, 1.0 - FUNNEL_RATE))
    global_survivors = ~global_reject
    local = np.full(n, np.nan)
    bids = np.full(n, np.nan)
    maxev = np.full(n, np.nan)
    pwin_best = np.full(n, np.nan)
    rng = np.random.default_rng(42)
    ref, tree, radius = build_tree(hist_pd, h)

    # Stage 1: global CatBoost PD is applied before any local matching.
    # Stage 2 local risk: calculate only for Stage 1 survivors; exclude each row itself.
    chunks = range(0, n, CHUNK_SIZE)
    iterator = tqdm(chunks, total=math.ceil(n / CHUNK_SIZE), desc=f'Backtest h={h:.3f}, delta={delta:+.0%}: local PD', unit='1%') if show_progress else chunks
    for start in iterator:
        stop = min(start + CHUNK_SIZE, n)
        query_rows = np.flatnonzero(global_survivors[start:stop]) + start
        if not len(query_rows):
            continue
        neighbors = kernel_neighbors(bt_pd[query_rows], ref, tree, radius, h)
        for row, (inds, weights) in zip(query_rows, neighbors):
            keep = inds != complete_idx[row]
            inds = inds[keep]
            weights = weights[keep]
            if len(inds):
                weights = weights / weights.sum()
                local[row] = float(weights @ hist_default[inds])
        del neighbors

    if not np.isfinite(local[global_survivors]).all():
        raise ValueError('Local PD missing for one or more Stage 1 Batch 2 survivors')
    risk_cutoff = float(np.quantile(local[global_survivors], 1.0 - FUNNEL_RATE))
    risk_reject = np.zeros(n, dtype=bool)
    risk_reject[global_survivors] = exact_top_fraction_mask(local[global_survivors], FUNNEL_RATE)
    local_survivors = global_survivors & ~risk_reject

    # Stage 3: calculate candidate rates and maximum expected profit only for Stage 2 survivors.
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

    remaining_applicants = local_survivors & np.isfinite(maxev)
    if not remaining_applicants.any():
        raise ValueError('No Stage 2 Batch 2 applicants had usable quotes for expected-profit calculation')
    profit_cutoff = max(0.0, float(np.quantile(maxev[remaining_applicants], 0.035)))
    print(f'Batch 2 Stage 3 expected-profit cutoff: €{profit_cutoff:,.2f}')
    profit_reject = np.zeros(n, dtype=bool)
    profit_reject[remaining_applicants] = maxev[remaining_applicants] <= profit_cutoff
    bids[~remaining_applicants | global_reject | risk_reject | profit_reject] = np.nan

    c1 = bt_c1 * (1.0 + delta); c2 = bt_c2 * (1.0 + delta); typ = bt.borrowertype.to_numpy(int)
    own_effective = bids.copy(); own_effective[typ == 3] -= TASTE_DISCOUNT
    rival1_effective = c1.copy(); rival1_effective[typ == 1] -= TASTE_DISCOUNT
    rival2_effective = c2.copy(); rival2_effective[typ == 2] -= TASTE_DISCOUNT
    won = np.isfinite(bids) & (own_effective < rival1_effective) & (own_effective < rival2_effective)
    realized = np.where(won, PRINCIPAL * ((1.0 - bt_default) * bids - bt_default), 0.0)
    d_rate = float(bt_default[won].mean()) if won.any() else np.nan
    return {'h': h, 'delta': delta, 'global_risk_cutoff': global_cutoff,
            'risk_cutoff': risk_cutoff, 'profit_cutoff': profit_cutoff,
            'n_quote_complete': n, 'n_stage1_rejected': int(global_reject.sum()),
            'n_stage2_rejected': int(risk_reject.sum()), 'n_stage3_rejected': int(profit_reject.sum()),
            'n_bids': int(np.isfinite(bids).sum()), 'n_won': int(won.sum()),
            'win_rate_among_complete': float(won.mean()), 'default_rate_won': d_rate,
            'total_profit_eur': float(realized.sum()),
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

# Update Batch 2 explanation, which now has all three filters.
nb['cells'][8]['source'] = '''## 4. Batch 2 Backtest: Three-stage 3.5% rejection funnel

On the quote-complete cohort, apply CatBoost global PD rejection first, compute leave-one-out local kernel PD only for global-risk survivors, then price only local-risk survivors. Stage 3 rejects applicants at or below the maximum of zero and the 3.5th-percentile expected-profit threshold.\n'''.splitlines(keepends=True)

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']), filename=f'{path}:cell{i}')
print('Updated Batch 2 to three-stage funnel; all code cells parse.')
