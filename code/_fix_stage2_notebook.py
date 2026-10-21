import ast
import json
from pathlib import Path

path = Path('code/02_risk_matching_optimizer.ipynb')
notebook = json.loads(path.read_text(encoding='utf-8'))
notebook['cells'][-1]['source'] = [
    "TARGET_PROFIT_EUR=30_900_000\n",
    "TARGET_DEFAULT_RATE=0.1225\n",
    "bt_profit=float(selected_backtest['total_profit_eur'])\n",
    "bt_default=float(selected_backtest['default_rate_won']) if np.isfinite(selected_backtest['default_rate_won']) else np.nan\n",
    "gate_report=pd.DataFrame([\n",
    " {'gate':'Backtest P&L exceeds €30.9M','observed':bt_profit,'threshold':TARGET_PROFIT_EUR,'pass':bt_profit>TARGET_PROFIT_EUR},\n",
    " {'gate':'Won-loan default rate below 12.25%','observed':bt_default,'threshold':TARGET_DEFAULT_RATE,'pass':np.isfinite(bt_default) and bt_default<TARGET_DEFAULT_RATE},\n",
    "])\n",
    "gate_report['sample_n']=[selected_backtest['n_quote_complete'],selected_backtest['n_won']]\n",
    "display(gate_report)\n",
    "print(f\"Benchmark covers {selected_backtest['n_quote_complete']:,} of 100,000 Batch 2 cases with complete rival quotes.\")\n",
    "\n",
    "final_bids[['id','rate']].to_csv(PROCESSED_DIR/'batch3_lender3_rates_stage2.csv',index=False)\n",
    "final_bids.to_csv(PROCESSED_DIR/'batch3_lender3_rates_stage2_audit.csv',index=False)\n",
    "backtest_summary.to_csv(PROCESSED_DIR/'stage2_bandwidth_backtest.csv',index=False)\n",
    "stress_summary.to_csv(PROCESSED_DIR/'stage2_competitor_stress_test.csv',index=False)\n",
    "gate_report.to_csv(PROCESSED_DIR/'stage2_backtest_gates.csv',index=False)\n",
    "print('Wrote Stage 2 bid CSV, audit, bandwidth, stress-test, and gate outputs.')",
]
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
for name in ['code/01_retrain_catboost.ipynb', 'code/02_risk_matching_optimizer.ipynb']:
    nb = json.loads(Path(name).read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
print('JSON valid; all code cells parse.')
