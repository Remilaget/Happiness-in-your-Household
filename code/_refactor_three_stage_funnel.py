import ast
import json
from pathlib import Path

path = Path('code/02_risk_matching_optimizer.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

# Replace Batch 3 load/merge cell so the original complete population and IDs are preserved.
nb['cells'][3]['source'] = '''batch2 = pd.read_csv(PROCESSED_DIR / 'Batch2.csv')
batch2_pd = pd.read_csv(PROCESSED_DIR / 'batch2_p_default_catboost.csv')
batch3 = pd.read_csv(RAW_DIR / 'NewApplications_Lender3_Round2.csv')
# Load initial Stage 2 PDs as requested; keep all original IDs for final reconstruction.
batch3_pd = pd.read_csv(PROCESSED_DIR / 'batch3_p_default_catboost.csv')
diag = pd.read_csv(RAW_DIR / 'Round1_Diagnostic_33.csv')

for name, frame in [('Batch2', batch2), ('Batch2 PD', batch2_pd), ('Batch3', batch3), ('Batch3 PD', batch3_pd), ('diagnostic', diag)]:
    assert frame.id.is_unique, f'{name} IDs are not unique'
assert all(len(x) == 100_000 for x in [batch2, batch2_pd, batch3, batch3_pd, diag])

hist = batch2.merge(batch2_pd[['id', 'pd']].rename(columns={'pd': 'p_hist'}), on='id', validate='one_to_one')
# Keep the unfiltered full population separate from the sequential working funnel.
original_ids = batch3[['id']].copy()
all_batch3 = batch3.merge(batch3_pd[['id', 'pd']].rename(columns={'pd': 'p_default'}), on='id', validate='one_to_one')
assert len(all_batch3) == 100_000 and all_batch3.id.is_unique
assert all_batch3.p_default.between(0, 1).all()

assert hist.p_hist.between(0, 1).all()
assert hist.default.isin([0, 1]).all() and hist.accepted.isin([0, 1]).all()
comp_mask = hist[['competing1', 'competing2']].notna().all(axis=1).to_numpy()
print(f'Batch 2: {len(hist):,} labeled; complete rival quotes: {comp_mask.sum():,} ({comp_mask.mean():.1%})')
print(f'Observed accepted/won rate={hist.accepted.mean():.2%}; default among accepted={hist.loc[hist.accepted.eq(1), "default"].mean():.2%}')
print(f'Observed Stage 1 diagnostic profit: €{hist.profit.sum():,.0f}')
'''.splitlines(keepends=True)

# Update section descriptions.
nb['cells'][12]['source'] = [
    "## 6. Sequential Batch 3 funnel: global risk, local risk, expected profit\n",
    "\n",
    "Each stage rejects exactly 3.5% of the applicants currently remaining (using rank-based selection so ties do not change the count). Stage 1 uses CatBoost PD and runs before any KD-tree work. Stage 2 computes local kernel PD only for Stage 1 survivors. Stage 3 prices only Stage 2 survivors, then removes applicants below the 3.5th percentile of their expected profits. The final result is reconstructed against all original IDs with `interest_rate = NaN` for every rejection.\n",
]

nb['cells'][13]['source'] = '''FUNNEL_RATE = 0.035

def exact_top_fraction_mask(values, fraction):
    """Select exactly floor(fraction*n) largest values, with stable tie handling."""
    values = np.asarray(values, dtype=float)
    n_reject = int(np.floor(fraction * len(values)))
    mask = np.zeros(len(values), dtype=bool)
    if n_reject:
        order = np.argsort(-values, kind='stable')
        mask[order[:n_reject]] = True
    return mask

# Stage 1: initial global CatBoost risk cut (before any local risk or rate work).
work = all_batch3[['id', 'p_default']].copy().reset_index(drop=True)
cutoff_stage1 = float(work.p_default.quantile(1.0 - FUNNEL_RATE))
reject1_mask = exact_top_fraction_mask(work.p_default.to_numpy(), FUNNEL_RATE)
rejected_stage1 = work.loc[reject1_mask, 'id'].copy()
work = work.loc[~reject1_mask].reset_index(drop=True)
print(f'Stage 1 global PD 96.5th percentile: {cutoff_stage1:.6f}; rejected {len(rejected_stage1):,}; remaining {len(work):,}')
assert len(rejected_stage1) == int(np.floor(FUNNEL_RATE * len(all_batch3)))

# Stage 2: local kernel risk only for Stage 1 survivors.
local_risk_stage2 = precompute_local_risk(
    work.p_default.to_numpy(), hist_pd, hist_default, SELECTED_H,
    show_progress=True, label='Stage 2 local risk (Stage 1 survivors)'
)
work['local_pd'] = local_risk_stage2
assert work.local_pd.notna().all(), 'Local risk missing for one or more Stage 1 survivors'
cutoff_stage2 = float(work.local_pd.quantile(1.0 - FUNNEL_RATE))
reject2_mask = exact_top_fraction_mask(work.local_pd.to_numpy(), FUNNEL_RATE)
rejected_stage2 = work.loc[reject2_mask, 'id'].copy()
work = work.loc[~reject2_mask].reset_index(drop=True)
print(f'Stage 2 local PD 96.5th percentile: {cutoff_stage2:.6f}; rejected {len(rejected_stage2):,}; remaining {len(work):,}')
assert len(rejected_stage2) == int(np.floor(FUNNEL_RATE * (len(all_batch3) - len(rejected_stage1))))

# Stage 3: expected profit and optimal rates only for Stage 2 survivors.
def price_survivors(survivors, h, show_progress=True):
    q = survivors.p_default.to_numpy(dtype=float)
    n = len(q)
    result_rate = np.full(n, np.nan)
    result_profit = np.full(n, np.nan)
    result_pwin = np.full(n, np.nan)
    rng = np.random.default_rng(42)
    ref, tree, radius = build_tree(hist_pd, h)
    chunks = range(0, n, CHUNK_SIZE)
    iterator = tqdm(chunks, total=math.ceil(n / CHUNK_SIZE), desc='Stage 3 pricing (survivors)', unit='1%') if show_progress else chunks
    for start in iterator:
        stop = min(start + CHUNK_SIZE, n)
        matches = kernel_neighbors(q[start:stop], ref, tree, radius, h)
        for offset, (inds, weights) in enumerate(matches):
            row = start + offset
            cm = comp_mask[inds]
            cinds = inds[cm]
            if not len(cinds):
                continue
            cw = weights[cm]
            cw = cw / cw.sum()
            rates = get_adaptive_rate_grid(hist_c1[cinds], hist_c2[cinds], n_points=N_RATES, rng=rng)
            pwin = p_win_by_rate(rates, hist_c1[cinds], hist_c2[cinds], cw)
            ev = expected_profit(rates, float(survivors.iloc[row].local_pd), pwin)
            best = int(np.argmax(ev))
            result_rate[row] = rates[best]
            result_profit[row] = ev[best]
            result_pwin[row] = pwin[best]
        del matches
    return pd.DataFrame({'id': survivors.id.to_numpy(),
                         'expected_profit': result_profit,
                         'optimal_interest_rate': result_rate,
                         'p_win_at_optimum': result_pwin})

priced_stage3 = price_survivors(work, SELECTED_H, show_progress=True)
work = work.merge(priced_stage3, on='id', how='left', validate='one_to_one')
assert work.expected_profit.notna().all(), 'Expected profit missing for one or more Stage 2 survivors'
cutoff_stage3 = float(work.expected_profit.quantile(FUNNEL_RATE))
print(f'Stage 3 expected-profit 3.5th percentile: €{cutoff_stage3:,.2f}')
if cutoff_stage3 > 100.0:
    print('⚠️ WARNING: 3.5% profit threshold is above 100€')

reject3_mask = work.expected_profit < cutoff_stage3
rejected_stage3 = work.loc[reject3_mask, 'id'].copy()
work_survivors = work.loc[~reject3_mask].copy()
print(f'Stage 3 rejected {len(rejected_stage3):,}; final survivors {len(work_survivors):,}')

# Reconstruct the full 100,000-ID result. Rejected IDs and any non-survivors have NaN interest_rate.
final_rates = work_survivors[['id', 'optimal_interest_rate']].rename(columns={'optimal_interest_rate': 'interest_rate'})
final_output = original_ids.merge(final_rates, on='id', how='left', validate='one_to_one')
final_output['interest_rate'] = final_output['interest_rate'].astype(float)

rejected_all = pd.Index(pd.concat([rejected_stage1, rejected_stage2, rejected_stage3], ignore_index=True))
assert len(final_output) == 100_000 and final_output.id.is_unique
assert set(final_output.id) == set(original_ids.id)
assert final_output.loc[final_output.id.isin(rejected_all), 'interest_rate'].isna().all()
assert final_output.loc[final_output.id.isin(work_survivors.id), 'interest_rate'].notna().all()
print(f'Final output rows: {len(final_output):,}; priced: {final_output.interest_rate.notna().sum():,}; blank/NaN rate: {final_output.interest_rate.isna().sum():,}')
print(f'Rejection counts — Stage 1: {len(rejected_stage1):,}; Stage 2: {len(rejected_stage2):,}; Stage 3: {len(rejected_stage3):,}')
'''.splitlines(keepends=True)

# Update plot cell for new names and thresholds.
nb['cells'][14]['source'] = '''# Funnel diagnostics: retained local-risk and expected-profit distributions, plus final rates.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(all_batch3.p_default, bins=50, color='#0072B2', alpha=0.55)
axes[0].axvline(cutoff_stage1, color='darkred', linestyle='--', label=f'Stage 1 cutoff={cutoff_stage1:.1%}')
axes[0].set_title('Initial CatBoost PD distribution'); axes[0].set_xlabel('CatBoost PD'); axes[0].legend(); axes[0].grid(alpha=.3)
axes[1].hist(work_survivors.local_pd, bins=50, color='#009E73', alpha=0.6)
axes[1].axvline(cutoff_stage2, color='darkred', linestyle='--', label=f'Stage 2 cutoff={cutoff_stage2:.1%}')
axes[1].set_title('Stage 2 survivor local PD'); axes[1].set_xlabel('Local kernel PD'); axes[1].legend(); axes[1].grid(alpha=.3)
axes[2].scatter(work_survivors.local_pd, work_survivors.optimal_interest_rate, alpha=.12, s=5, color='#CC79A7')
axes[2].set_title('Final rates by local risk'); axes[2].set_xlabel('Local kernel PD'); axes[2].set_ylabel('Interest rate'); axes[2].grid(alpha=.3)
plt.tight_layout(); plt.show()
'''.splitlines(keepends=True)

# Change final export to use the all-ID reconstructed output.
old = "final_bids[['id','rate']].to_csv(PROCESSED_DIR/'batch3_lender3_rates_stage2.csv',index=False)\nfinal_bids.to_csv(PROCESSED_DIR/'batch3_lender3_rates_stage2_audit.csv',index=False)"
new = "final_output.rename(columns={'interest_rate':'rate'}).to_csv(PROCESSED_DIR/'batch3_lender3_rates_stage2.csv',index=False)\nfinal_output.to_csv(PROCESSED_DIR/'batch3_lender3_rates_stage2_audit.csv',index=False)"
nb['cells'][16]['source'] = ''.join(nb['cells'][16]['source']).replace(old, new).splitlines(keepends=True)

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']), filename=f'{path}:cell{i}')
print('Three-stage funnel implemented; every code cell parses.')
