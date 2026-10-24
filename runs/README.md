# runs/ — the two runnable pipeline notebooks

Each notebook sits in its own folder so you can open and run whichever one you
want without the other in the way. Both are plain Jupyter notebooks (not
Streamlit — `streamlit run streamlit_dashboard/app.py` only serves the website,
which loads whatever artifacts already sit in `models/v2_static/`).

| Folder            | Notebook                                                          | Writes model artifacts to          | Use it when                                                                                                                                                                                       |
| ----------------- | ----------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `static/`         | `LSTM_XGBoost_Ensemble_Precipitation_STATIC.ipynb`                | `models/v2_static/`                | This is the real, locked-configuration pipeline. Re-running it **overwrites the artifacts the Streamlit site loads.**                                                                             |
| `xgb_experiment/` | `LSTM_XGBoost_Ensemble_Precipitation_STATIC_xgb_experiment.ipynb` | `models/v2_static_xgb_experiment/` | Sandbox copy for diagnosing the XGBoost / ensemble under-prediction issue. Same pipeline plus extra diagnostic cells; the redirected output dir means it **cannot clobber** the site's artifacts. |
| `multistep_7day/` | `LSTM_XGBoost_Ensemble_Precipitation_7DAY.ipynb` | none — evaluation only | Loads the already-trained `models/v2_static/` artifacts (no retraining) and rolls them forward **recursively** to a 7-day horizon, to test the panel's "forecast further ahead" suggestion with evidence. Use it to see why that suggestion was not adopted for the main deliverable: the recursive forecast collapses to a near-constant value almost immediately instead of tracking real rainfall, so it does not fix the 1-day underprediction problem — it just repeats it. |
| `retuned/` | `LSTM_XGBoost_Ensemble_Precipitation_RETUNED.ipynb` | `models/v2_retuned/` | Same pipeline and same step order as `static/`, with only the **XGBoost** values open to change. The LSTM, the shared preprocessing values and the ensemble weighting rule are declared frozen and the notebook raises if any of them differs. Adds evaluation the other runs do not have: a predictability ceiling, event-detection scores (POD/FAR/CSI/HSS/AUC), skill scores against reference forecasts, and accumulated seven-day totals. Writes to its own directory, and hashes `models/v2_static/` on entry and exit so the artifacts the Streamlit site loads cannot be touched. |

## How to run either one

Both expect the training CSV `2619492.csv` and the output `models/` dir at the
**repo root**. The first code cell in each notebook (`# Path fix …`) walks up
from wherever the notebook is and `chdir`s to the repo root, so paths resolve
exactly as they did before the move. It is safe to re-run.

- **VS Code:** open the `.ipynb`, pick the `.venv` kernel (top-right), Run All.
- **CLI:**
  ```powershell
  .venv\Scripts\python -m pip install jupyter -r streamlit_dashboard\requirements.txt
  .venv\Scripts\jupyter notebook runs\static\LSTM_XGBoost_Ensemble_Precipitation_STATIC.ipynb
  ```
- **Headless:**
  ```powershell
  .venv\Scripts\jupyter nbconvert --to notebook --execute --inplace runs\static\LSTM_XGBoost_Ensemble_Precipitation_STATIC.ipynb
  ```

Training the LSTM (100 epochs, CPU) takes a few minutes. After the `static/`
notebook finishes, restart the Streamlit app to pick up the refreshed
`models/v2_static/` files.

## Older notebooks

The earlier exploratory notebooks (REFACTORED, ADAM, residuals, test, Piers,
XGBoost-only, …) were moved to `../archive/`. Nothing in the current pipeline or
the Streamlit site depends on them.

streamlit run streamlit_dashboard/app.py

$env:PRECIP_MODEL_DIR = "v2_static_xgb_experiment"; streamlit run streamlit_dashboard/app.py
