# Customer Segmentation & Sales Prediction

A complete ML project: a Colab notebook with the nine lab experiments and a Streamlit dashboard backed by saved scikit-learn models.

## What was built
The raw data is transaction-level with no ready-made target, so it is aggregated to **one row per customer** at a snapshot date (2011-09-11). Features use only earlier transactions; targets use the following 90 days (no leakage):

| Task | Target | Best model (test) |
|---|---|---|
| Regression | `FutureSpend` - GBP spent in the next 90 days | Hist Gradient Boosting, RMSE ~ GBP 988, R2 0.62 (log-scale R2 0.38) |
| Classification | `Active` - buys again within 90 days (44% do) | Tuned Random Forest, accuracy 75%, ROC-AUC 0.82 |
| Clustering | RFM segments (no target) | K-Means, k = 4, silhouette 0.38 |

## Files
| File | Purpose |
|---|---|
| `Customer_Segmentation_Sales_Prediction.ipynb` | The 9 experiments (already executed, outputs included) |
| `Colab_Launch_Streamlit.ipynb` | Separate launcher: installs, trains if needed, starts Streamlit + public tunnel |
| `app.py` | Streamlit UI (7 pages) |
| `backend.py` | The single backend: cleaning, feature engineering, pipelines, training, segmentation, prediction. Also the training CLI: `python backend.py [dataset]` |
| `data/` | Put the dataset here (`online_retail_II.xlsx` included) |
| `models/` | Saved models, segmenter, metrics, dashboard aggregates |
| `requirements.txt` | Dependencies |

## A. Run the notebook in Google Colab
1. Go to https://colab.research.google.com -> **File -> Upload notebook** -> choose `Customer_Segmentation_Sales_Prediction.ipynb`.
2. **Runtime -> Run all**. The first cell asks you to upload `customer_segmentation_project.zip` (this project). It unzips it and uses the included dataset. (If the dataset is missing it asks you to upload it - CSV or Excel.)
3. Reading the 1M-row Excel file takes about a minute; the whole notebook runs in roughly 4-5 minutes on a standard runtime.

## B. Run the Streamlit app in Google Colab
1. Upload `Colab_Launch_Streamlit.ipynb` to Colab (separate notebook).
2. Run the cells in order: upload the ZIP -> (optional) replace the dataset -> install -> check/retrain models -> launch.
3. The last cell prints a `https://xxxx.trycloudflare.com` URL; open it. **Leave that cell running.**
4. Notes: the URL changes on every run; free Colab runtimes disconnect when idle and after ~12 h, so everything is temporary. A LocalTunnel fallback and a troubleshooting table are in the launcher notebook.

## C. Run locally (Mac/Linux/Windows)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py          # models are already saved in models/
# optional: python backend.py   (re-train, ~3 min)
```

## D. Where to put your dataset
* Local: copy it to `data/online_retail_II.xlsx` (or pass any path: `python backend.py data/my_file.csv`), then restart the app.
* Colab: in the launcher set `REPLACE_DATA = True` in Step 2 and upload the file; models are retrained for it.
* Required columns: `Invoice, StockCode, Quantity, InvoiceDate, Price, Customer ID, Country` (`Description` optional); CSV or Excel (all sheets are stacked).
* In the app, *Customer Segmentation -> Batch-score your own transaction file* scores an uploaded file with the saved models without retraining.

## Design notes and limitations
* Preprocessing (median impute -> log1p -> standardise, one-hot country) is inside the scikit-learn pipelines, so training and inference are identical and the scaler is fitted on training data only.
* The same split (80/20, stratified, seed 42) is used by the notebook and `backend.py`, so numbers match; the notebook's last section verifies the saved models reproduce the dashboard's metrics.
* Customer spend is extremely skewed: predictions are good for ranking customers and for totals, not exact per-customer forecasts. Linear models show negative R2 in GBP (but positive on the log scale) for this reason.
* Single snapshot split (no rolling back-test); ~22% of lines have no Customer ID and are excluded; saved models are tied to the scikit-learn version (Colab retrains automatically).
