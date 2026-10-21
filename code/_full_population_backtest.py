import ast, json
from pathlib import Path

path = Path('code/02_risk_matching_optimizer.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

# Full 100k applicant arrays; only the reference pool is restricted to quote-complete rows.
nb['cells'][9]['source'] = '''hist_pd = hist.p_hist.to_numpy(float)
hist_default = hist.default.to_numpy(int)
hist_c1 = hist.competing1.to_numpy(float)
hist_c2 = hist.competing2.to_numpy(float)
bt = hist.reset_index(drop=True)
bt_pd = hist_pd
bt_global_pd = bt.p_hist.to_numpy(float)
bt_default = hist_default
bt_c1 = hist_c1
bt_c2 = hist_c2
quote_reference_idx = np.flatnonzero(comp_mask)

# comp_mask limits rival quote references and realized replay metrics; query population remains all 100,000.
def exact_top_fraction_mask(values, fraction):
    values = np.asarray(values, dtype=float)
    n_reject = int(np.floor(fraction * len(values)))
    mask = np.zeros(len(values), dtype=bool)
    if n_reject:
        mask[np.argsort(-values, kind='stable')[:n_reject]] = True
    return mask

def run_backtest(h, delta=0.0, show_progress=True):
    n = len(hist)
    if n != len(hist_pd) or n != len(comp_mask):
        raise ValueError('hist, hist_pd, and comp_mask must have matching full-population lengths')
    global_reject = exact_top_fraction_mask(hist_pd, FUNNEL_RATE)
    global_cutoff = float(np.quantile(hist_pd, 1.0 - FUNNEL_RATE))
    global_survivors = ~global_reject
    local = np.full(n, np.nan)
    bids = np.full(n, np.nan)
    maxev = np.full(n, np.nan)
    pwin_best = np.full(n, np.nan)
    rng = np.random.default_rng(42)
    ref = hist_pd[quote_reference_idx]
    ref_default = hist_default[quote_reference_idx]
    ref, tree, radius = build_tree(ref, h)

    # Stage 1: global PD cut over all Batch 2 applicants.
    # Stage 2: leave-one-out local PD only for Stage 1 survivors.
    chunks = range(0, n, CHUNK_SIZE)
    iterator = tqdm(chunks, total=math.ceil(n / CHUNK_SIZE), desc=f'Backtest h={h:.3f}, delta={delta:+.0%}: local PD', unit='1%') if show_progress else chunks
    for start in iterator:
        stop = min(start + CHUNK_SIZE, n)
        query_rows = np.flatnonzero(global_survivors[start:stop]) + start
        if not len(query_rows):
            continue
        neighbors = kernel_neighbors(hist_pd[query_rows], ref, tree, radius, h)
        for row, (ref_local_idx, weights) in zip(query_rows, neighbors):
            keep = quote_reference_idx[ref_local_idx] != row
            ref_local_idx = ref_local_idx[keep]
            weights = weights[keep]
            if len(ref_local_idx):
                weights = weights / weights.sum()
                local[row] = float(weights @ ref_default[ref_local_idx])
        del neighbors

    if not np.isfinite(local[global_survivors]).all():
        raise ValueError('Local PD missing for one or more Stage 1 Batch 2 survivors')
    risk_cutoff = float(np.quantile(local[global_survivors], 1.0 - FUNNEL_RATE))
    risk_reject = np.zeros(n, dtype=bool)
    risk_reject[global_survivors] = exact_top_fraction_mask(local[global_survivors], FUNNEL_RATE)
    local_survivors = global_survivors & ~risk_reject

    # Stage 3: price Stage 2 survivors; neighbor quotes come only from quote-complete references.
    chunks = range(0, n, CHUNK_SIZE)
    iterator = tqdm(chunks, total=math.ceil(n / CHUNK_SIZE), desc=f'Backtest h={h:.3f}, delta={delta:+.0%}: pricing', unit='1%') if show_progress else chunks
    for start in iterator:
        stop = min(start + CHUNK_SIZE, n)
        query_rows = np.flatnonzero(local_survivors[start:stop]) + start
        if not len(query_rows):
            continue
        neighbors = kernel_neighbors(hist_pd[query_rows], ref, tree, radius, h)
        for row, (ref_local_idx, weights) in zip(query_rows, neighbors):
            ref_rows = quote_reference_idx[ref_local_idx]
            # Exclude self from historical replay if the query itself is in the reference pool.
            keep = ref_rows != row
            ref_rows = ref_rows[keep]
            weights = weights[keep]
            if not len(ref_rows):
                continue
            rates = get_adaptive_rate_grid(hist_c1[ref_rows], hist_c2[ref_rows], n_points=N_RATES, rng=rng)
            pwin = scenario_pwin_exact(rates, hist_c1[row], hist_c2[row], delta=delta)
            ev = expected_profit(rates, local[row], pwin)
            best = int(np.argmax(ev))
            maxev[row] = ev[best]
            pwin_best[row] = pwin[best]
            bids[row] = rates[best]
        del neighbors

    priced_survivors = local_survivors & np.isfinite(maxev)
    if not priced_survivors.any():
        raise ValueError('No Stage 2 Batch 2 applicants had usable quote-complete references for pricing')
    profit_cutoff = max(0.0, float(np.quantile(maxev[priced_survivors], 0.035)))
    print(f'Stage 3 expected-profit cutoff applied: €{profit_cutoff:.2f}')
    profit_reject = priced_survivors & (maxev <= profit_cutoff)
    bids[~priced_survivors | global_reject | risk_reject | profit_reject] = np.nan

    # Backtest outcomes are observable only for query applicants with both rival quotes.
    quote_complete = comp_mask
    c1 = hist_c1[quote_complete] * (1.0 + delta)
    c2 = hist_c2[quote_complete] * (1.0 + delta)
    typ = bt.borrowertype.to_numpy(int)[quote_complete]
    bids_complete = bids[quote_complete]
    own_effective = bids_complete.copy(); own_effective[typ == 3] -= TASTE_DISCOUNT
    rival1_effective = c1.copy(); rival1_effective[typ == 1] -= TASTE_DISCOUNT
    rival2_effective = c2.copy(); rival2_effective[typ == 2] -= TASTE_DISCOUNT
    won_complete = np.isfinite(bids_complete) & (own_effective < rival1_effective) & (own_effective < rival2_effective)
    defaults_complete = bt_default[quote_complete]
    realized = np.where(won_complete, PRINCIPAL * ((1.0 - defaults_complete) * bids_complete - defaults_complete), 0.0)
    d_rate = float(defaults_complete[won_complete].mean()) if won_complete.any() else np.nan

    return {'h': h, 'delta': delta, 'global_risk_cutoff': global_cutoff,
            'risk_cutoff': risk_cutoff, 'profit_cutoff': profit_cutoff,
            'n_applicants': n, 'n_quote_complete': int(quote_complete.sum()),
            'n_stage1_rejected': int(global_reject.sum()),
            'n_stage2_rejected': int(risk_reject.sum()),
            'n_stage3_rejected': int(profit_reject.sum()),
            'n_priced_full_cohort': int(np.isfinite(bids).sum()),
            'n_bids': int(np.isfinite(bids_complete).sum()),
            'n_won': int(won_complete.sum()),
            'win_rate_among_complete': float(won_complete.mean()),
            'default_rate_won': d_rate, 'total_profit_eur': float(realized.sum()),
            'mean_rate_won': float(bids_complete[won_complete].mean()) if won_complete.any() else np.nan,
            'bids': bids, 'local_risk': local, 'expected_profit': maxev,
            'p_win_at_optimum': pwin_best, 'won': won_complete}

backtests = []
for h in H_GRID:
    result = run_backtest(h)
    backtests.append(result)
    print({k: v for k, v in result.items() if k not in {'bids', 'local_risk', 'expected_profit', 'p_win_at_optimum', 'won'}})
backtest_summary = pd.DataFrame([{k: v for k, v in r.items() if k not in {'bids', 'local_risk', 'expected_profit', 'p_win_at_optimum', 'won'}} for r in backtests])
display(backtest_summary)
'''.splitlines(keepends=True)

nb['cells'][8]['source'] = '''## 4. Batch 2 Backtest: Full-cohort three-stage funnel

Run all 100,000 applicants through the CatBoost global-risk, local-risk, and profitability stages. Build local pricing references only from applicants with complete rival quotes; calculate realized win and P&L metrics only on those query rows as well.\n'''.splitlines(keepends=True)

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']), filename=f'{path}:cell{i}')
print('Full-population Batch 2 backtest implemented; all code cells parse.')
