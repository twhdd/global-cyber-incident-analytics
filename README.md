# Global Cyber Incident Intelligence Dashboard

Python + Streamlit dashboard for the DA2 project. R remains the source of truth for data cleaning, feature engineering, training, tuning, and evaluation. This starter reads the existing MySQL incident table and does not retrain models.

## Included
- **Overview:** KPIs, incident activity over time, severity, incident types, geographic counts, dataset health.
- **Incident Explorer:** year/severity/type/country filters, text search, table, CSV export.

## 1. Open in VS Code
Extract the ZIP and open this folder in VS Code. In its terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked, run `\.venv\Scripts\python.exe -m pip install -r requirements.txt` instead.

## 2. Configure MySQL
Copy `.env.example` to `.env` and enter the credentials used by your local MySQL server. Defaults target `127.0.0.1:3306`, database `cyber_incidents`, table `cyber_incident_data`. Do not commit `.env` or secrets.

The app automatically loads the project `.env` file. Alternatively, you can use `.streamlit/secrets.toml` as shown below.

Example `.streamlit/secrets.toml`:

```toml
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_USER = "root"
DB_PASSWORD = "your-local-password"
DB_NAME = "cyber_incidents"
DB_TABLE = "cyber_incident_data"
```

## 3. Run

```powershell
streamlit run app.py
```

Usually opens at `http://localhost:8501`.

## 4. If fields aren't detected
Expand **Detected column mapping** on Overview or inspect columns on Incident Explorer. Add the exact MySQL column names to `ALIASES` in `src/schema.py`. The app does not fabricate missing geography or severity.

SQL schema check:

```sql
USE cyber_incidents;
DESCRIBE cyber_incident_data;
SELECT COUNT(*) FROM cyber_incident_data;
```

## Next phase
Place existing R-generated result CSVs in `data/r_outputs/` when implementing Model Performance and Feature Insights. Do not rerun model training just to display those results. This starter uses `SELECT` queries only and caches the loaded table for five minutes.


## Confirmed MySQL schema (checked by the project owner)

The live `cyber_incidents.cyber_incident_data` table returned 3,414 rows and 55 columns. The initial dashboard maps these real columns:

- `incident_year` — yearly activity
- `incident_type` — incident type distribution
- `initiator_country` — country distribution of recorded initiators (not victim/receiver geography)
- `severity_class` — Low / Medium / High severity

The current live severity counts were Low 1,783, Medium 1,410, High 221. The final deduplicated DA2 modelling dataset has 3,412 rows, so do not present that modelling-stage count as the live SQL table count.

The app now loads `.env` using `python-dotenv`. Keep `.env` private and never commit it.


## Model and feature pages

The starter ZIP includes selected CSV outputs copied from the user's DA2 archive into `data/r_outputs/`. These are existing R results, not regenerated outputs:

- `cv_12_model_results.csv`
- `tuned_test_results.csv`
- `best_hyperparameters.csv`
- `cross_model_feature_importance.csv`
- `all_model_feature_importance.csv`

The two added pages read these CSVs locally. Keep the original R pipeline unchanged; if the R outputs are updated later, replace these CSVs with the latest verified outputs.


## Severity prediction demo (requires local R)

The prediction page uses the saved `DA2_Tuned_Model_Objects.rds` file and `prediction_bridge.R` to call the original fitted R workflow. It does not use a Python approximation and does not retrain models.

Requirements:
- R installed on the same computer as Streamlit.
- The R packages used by the DA2 tuning workflow installed in that R library, including `tidymodels` and the model-engine packages used by the saved fits.
- `Rscript.exe` discoverable on PATH, or set `RSCRIPT_PATH` in `.env` to the full path to `Rscript.exe`.

Example `.env` addition (change the version/path to the one actually installed):
```dotenv
RSCRIPT_PATH=C:\Program Files\R\R-4.5.1\bin\Rscript.exe
```

The demo selects an existing MySQL record as a baseline so the model receives all 54 predictors expected by the fitted preprocessing workflow. Users can change selected fields; the remaining fields stay inherited from the baseline row. The target `severity_class` is removed before inference to prevent target leakage. Predictions are academic estimates, not causal or operational verdicts.
