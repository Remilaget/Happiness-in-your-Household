import ast, json
from pathlib import Path

path = Path('code/02_risk_matching_optimizer.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

# Replace only the adaptive grid helper, preserving the remaining market/pricing helpers.
s = ''.join(nb['cells'][5]['source'])
start = s.index('def get_adaptive_rate_grid(')
end = s.index('\ndef scenario_pwin_exact', start)
s = s[:start] + '''def get_adaptive_rate_grid(rival1, rival2, p_local, n_points=10, buffer=0.01, rng=None):
    """Return applicant rate grid bounded below by breakeven and a toxic-market flag."""
    r1 = np.asarray(rival1, dtype=float)
    r2 = np.asarray(rival2, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    r1 = r1[np.isfinite(r1)]; r2 = r2[np.isfinite(r2)]
    r1_sample = rng.choice(r1, size=max(1, int(np.sqrt(len(r1)))), replace=False) if len(r1) else np.array([])
    r2_sample = rng.choice(r2, size=max(1, int(np.sqrt(len(r2)))), replace=False) if len(r2) else np.array([])
    sampled = np.concatenate([r1_sample, r2_sample])
    breakeven = p_local / (1.0 - p_local)
    if not len(sampled):
        r_min = breakeven
        r_max = min(1.0, r_min + 0.05)
        return np.linspace(r_min, r_max, n_points), False

    c_max = float(sampled.max())
    market_below_breakeven = bool(breakeven > c_max)
    r_min = max(breakeven, float(sampled.min()) - buffer)
    r_max = min(1.0, max(c_max + buffer, r_min + 0.05))
    return np.linspace(r_min, r_max, n_points), market_below_breakeven
''' + s[end:]
nb['cells'][5]['source'] = s.splitlines(keepends=True)

# Patch Batch 2 backtest pricing loop; keep its Stage 3 cutoff exactly as implemented.
s = ''.join(nb['cells'][9]['source'])
s = s.replace('''    # Stage 3: price Stage 2 survivors; neighbor quotes come only from quote-complete references.
    chunks = range(0, n, CHUNK_SIZE)''', '''    # Stage 3: price Stage 2 survivors; neighbor quotes come only from quote-complete references.
    toxic_market_count = 0
    chunks = range(0, n, CHUNK_SIZE)''')
s = s.replace('''            rates = get_adaptive_rate_grid(hist_c1[ref_rows], hist_c2[ref_rows], n_points=N_RATES, rng=rng)
            pwin = scenario_pwin_exact(rates, hist_c1[row], hist_c2[row], delta=delta)''', '''            rates, market_below_breakeven = get_adaptive_rate_grid(
                hist_c1[ref_rows], hist_c2[ref_rows], local[row], n_points=N_RATES, rng=rng
            )
            if market_below_breakeven:
                toxic_market_count += 1
            pwin = scenario_pwin_exact(rates, hist_c1[row], hist_c2[row], delta=delta)''')
s = s.replace('''        del neighbors

    priced_survivors = local_survivors & np.isfinite(maxev)''', '''        del neighbors
    print(f"Diagnostic: {toxic_market_count} applicants had competitor maximum rates strictly below our breakeven floor.")

    priced_survivors = local_survivors & np.isfinite(maxev)''')
nb['cells'][9]['source'] = s.splitlines(keepends=True)

# Patch Batch 3 pricing loop. Keep its existing stage-3 profit cutoff untouched.
s = ''.join(nb['cells'][13]['source'])
s = s.replace('''    rng = np.random.default_rng(42)
    ref, tree, radius = build_tree(hist_pd, h)
    chunks = range(0, n, CHUNK_SIZE)''', '''    rng = np.random.default_rng(42)
    toxic_market_count = 0
    ref, tree, radius = build_tree(hist_pd, h)
    chunks = range(0, n, CHUNK_SIZE)''')
s = s.replace('''            rates = get_adaptive_rate_grid(hist_c1[cinds], hist_c2[cinds], n_points=N_RATES, rng=rng)
            pwin = p_win_by_rate(rates, hist_c1[cinds], hist_c2[cinds], cw)''', '''            local_pd = float(survivors.iloc[row].local_pd)
            rates, market_below_breakeven = get_adaptive_rate_grid(
                hist_c1[cinds], hist_c2[cinds], local_pd, n_points=N_RATES, rng=rng
            )
            if market_below_breakeven:
                toxic_market_count += 1
            pwin = p_win_by_rate(rates, hist_c1[cinds], hist_c2[cinds], cw)''')
s = s.replace('''            ev = expected_profit(rates, float(survivors.iloc[row].local_pd), pwin)''', '''            ev = expected_profit(rates, local_pd, pwin)''')
s = s.replace('''        del matches
    return pd.DataFrame({'id': survivors.id.to_numpy(),''', '''        del matches
    print(f"Diagnostic: {toxic_market_count} applicants had competitor maximum rates strictly below our breakeven floor.")
    return pd.DataFrame({'id': survivors.id.to_numpy(),''')
nb['cells'][13]['source'] = s.splitlines(keepends=True)

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']), filename=f'{path}:cell{i}')
print('Breakeven grid and toxic-market diagnostics applied; all code cells parse.')
