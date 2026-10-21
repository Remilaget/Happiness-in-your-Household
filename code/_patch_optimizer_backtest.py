import ast, json
from pathlib import Path

path=Path('code/02_risk_matching_optimizer.ipynb')
nb=json.loads(path.read_text(encoding='utf-8'))
cell=nb['cells'][5]
s=''.join(cell['source'])
needle='def expected_profit(rate_grid, p_local, p_win):'
insert='''def scenario_pwin_exact(rate, rival1, rival2, delta=0.0):
    """P(Win) for one specific rival pair under equal borrower preference types."""
    c1=np.asarray(rival1,dtype=float)*(1+delta)
    c2=np.asarray(rival2,dtype=float)*(1+delta)
    r=np.asarray(rate,dtype=float)
    win1=(r < c1-TASTE_DISCOUNT) & (r < c2)
    win2=(r < c1) & (r < c2-TASTE_DISCOUNT)
    win3=(r-TASTE_DISCOUNT < c1) & (r-TASTE_DISCOUNT < c2)
    return (win1.astype(float)+win2.astype(float)+win3.astype(float))/3.0

def expected_profit(rate_grid, p_local, p_win):'''
s=s.replace(needle,insert)
# Replace backtest local competitor-neighborhood Pwin / EV section with exact query quote pair.
s=s.replace("""            cm=comp_mask[inds]; cinds=inds[cm]; cw=w[cm]
            if len(cinds)==0: continue
            cw=cw/cw.sum()
            pw=p_win_by_rate(RATE_GRID,hist_c1[cinds],hist_c2[cinds],cw,delta=delta)
            ev=expected_profit(RATE_GRID,local[row],pw); j=int(np.argmax(ev))
            maxev[row]=ev[j]; pwin_best[row]=pw[j]
""","""            # Backtest knows this query's realized rival quotes; average its three preference outcomes exactly.
            pw=scenario_pwin_exact(RATE_GRID,bt_c1[row],bt_c2[row],delta=delta)
            ev=expected_profit(RATE_GRID,local[row],pw); j=int(np.argmax(ev))
            maxev[row]=ev[j]; pwin_best[row]=pw[j]
""")
# Replace realized P&L block with diagnostic observed acceptance for actual Stage 1 offer and exact proposed bids under preference rule.
s=s.replace("""    c1=bt_c1*(1+delta); c2=bt_c2*(1+delta); typ=bt.borrowertype.to_numpy(int)
    own_effective=bids.copy(); own_effective[typ==3]-=TASTE_DISCOUNT
    rival1_effective=c1.copy(); rival1_effective[typ==1]-=TASTE_DISCOUNT
    rival2_effective=c2.copy(); rival2_effective[typ==2]-=TASTE_DISCOUNT
    won=np.isfinite(bids)&(own_effective<rival1_effective)&(own_effective<rival2_effective)
""","""    c1=bt_c1*(1+delta); c2=bt_c2*(1+delta); typ=bt.borrowertype.to_numpy(int)
    own_effective=bids.copy(); own_effective[typ==3]-=TASTE_DISCOUNT
    rival1_effective=c1.copy(); rival1_effective[typ==1]-=TASTE_DISCOUNT
    rival2_effective=c2.copy(); rival2_effective[typ==2]-=TASTE_DISCOUNT
    won=np.isfinite(bids)&(own_effective<rival1_effective)&(own_effective<rival2_effective)
""")
nb['cells'][5]['source']=s.splitlines(keepends=True)
path.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf-8')
for c in nb['cells']:
    if c['cell_type']=='code': ast.parse(''.join(c['source']))
print('Exact-pair backtest patched; all code cells parse.')
