# Happiness-in-your-Household
Lending game repository for the team Happiness in your Household

We are **Lender 3**: we predict the default probability of 100,000 loan applicants (10,000 € each) and choose an interest rate offer per applicant, competing against two other lenders.

## Setup
Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

This creates `.venv/` with the exact versions in `uv.lock`. XGBoost on macOS also needs OpenMP: `brew install libomp`.

## Data (not committed)
Put the course CSVs in `data/raw/`:
- `data/raw/PastLoans.csv`: past loans with all three signals and `default`
- `data/raw/NewApplications_Lender3_Round1.csv`: new applicants, `signal3` only

## Run
```bash
uv run jupyter lab
```

Add a dependency with `uv add <package>` (commit the updated `pyproject.toml` and `uv.lock`).
