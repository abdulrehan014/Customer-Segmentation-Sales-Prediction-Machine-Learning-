"""
backend.py - data preparation, feature engineering, model training, segmentation
and prediction logic shared by the Colab notebook, the training CLI and the
Streamlit app (app.py).

Design in one paragraph
-----------------------
The raw dataset (UCI "Online Retail II") is transaction-level: one row per product
line on an invoice. We aggregate it to ONE ROW PER CUSTOMER as of a snapshot date
("cutoff"). All input features use only transactions BEFORE the cutoff; all targets use
only transactions in the HORIZON_DAYS AFTER the cutoff. That separation prevents target
leakage by construction:

    regression target      FutureSpend  = money spent in the next 90 days (GBP, 0 if none)
    classification target  Active       = 1 if the customer bought again in the next 90 days
    clustering             RFM (Recency, Frequency, Monetary) as of the cutoff, no target used
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, KFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
DEFAULT_DATA_FILE = DATA_DIR / "online_retail_II.xlsx"

# --------------------------------------------------------------------------- constants
RANDOM_STATE = 42
HORIZON_DAYS = 90
# Lines that are not merchandise (postage, fees, manual adjustments, vouchers...)
NON_PRODUCT_CODES = {"POST", "DOT", "M", "BANK CHARGES", "ADJUST", "ADJUST2", "AMAZONFEE",
                     "D", "S", "PADS", "C2", "CRUK", "B", "TEST001", "TEST002", "SP1002"}
TOP_COUNTRIES = ["United Kingdom", "EIRE", "Germany", "France"]
NUM_FEATURES = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "TotalQuantity",
                "UniqueProducts", "AvgLinesPerOrder", "TenureDays", "AvgDaysBetweenOrders",
                "Orders_Last90", "Spend_Last90", "ReturnRate"]
CAT_FEATURES = ["CountryGroup"]
FEATURES = NUM_FEATURES + CAT_FEATURES
CLUSTER_FEATURES = ["Recency", "Frequency", "Monetary"]
FEATURE_HELP = {
    "Recency": "Days since the customer's last purchase (before the snapshot date)",
    "Frequency": "Number of distinct invoices (orders)",
    "Monetary": "Total spend in GBP (excluding returns)",
    "AvgOrderValue": "Monetary / Frequency",
    "TotalQuantity": "Total units bought",
    "UniqueProducts": "Number of distinct products bought",
    "AvgLinesPerOrder": "Average product lines per order",
    "TenureDays": "Days between first purchase and the snapshot date",
    "AvgDaysBetweenOrders": "Average gap between orders (blank for one-order customers)",
    "Orders_Last90": "Orders placed in the 90 days before the snapshot",
    "Spend_Last90": "GBP spent in the 90 days before the snapshot",
    "ReturnRate": "Share of spend that was returned/cancelled (0-1)",
    "CountryGroup": "Customer country (top markets, others grouped)",
}


# =========================================================================== 1. LOADING / CLEANING
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column-name variants to the canonical schema and validate it."""
    ren = {"Customer ID": "CustomerID", "Customer_ID": "CustomerID", "customer_id": "CustomerID",
           "InvoiceNo": "Invoice", "UnitPrice": "Price", "Unit Price": "Price"}
    df = df.rename(columns={c: ren.get(str(c).strip(), str(c).strip()) for c in df.columns})
    need = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {missing}. "
                         f"Expected columns: {need} (+ optional Description).")
    if "Description" not in df.columns:
        df["Description"] = np.nan
    return df


def load_raw(path_or_buffer, name: str | None = None) -> pd.DataFrame:
    """Read a CSV or Excel transaction file (all sheets of an Excel file are stacked)."""
    name = (name or str(getattr(path_or_buffer, "name", path_or_buffer))).lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        sheets = pd.read_excel(path_or_buffer, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
    elif name.endswith(".csv"):
        try:
            df = pd.read_csv(path_or_buffer)
        except UnicodeDecodeError:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            df = pd.read_csv(path_or_buffer, encoding="ISO-8859-1")
    else:
        raise ValueError("Unsupported file type - please provide a .csv or .xlsx file.")
    return standardize_columns(df)


def load_raw_cached(path=DEFAULT_DATA_FILE) -> pd.DataFrame:
    """Reading the 1M-row Excel file takes ~1 minute; cache it as parquet next to the data."""
    path = Path(path)
    cache = PROC_DIR / f"raw_{path.stem}_{path.stat().st_size}.parquet"   # keyed by file name + size
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass
    raw = load_raw(path)
    try:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        raw.assign(Invoice=raw["Invoice"].astype(str), StockCode=raw["StockCode"].astype(str),
                   Description=raw["Description"].astype("object").where(raw["Description"].notna(), None).map(lambda v: v if v is None else str(v)),
                   CustomerID=raw["CustomerID"].astype("float64"),
                   Country=raw["Country"].astype(str)).to_parquet(cache)
    except Exception as e:  # caching is an optimisation only
        print(f"(parquet cache not written: {e})")
    return raw


def clean_transactions(raw: pd.DataFrame):
    """Return (sales, returns, log). `sales` = valid purchases, `returns` = cancellations."""
    df = standardize_columns(raw.copy())
    log = {"raw_rows": len(df)}
    n = len(df)
    df = df.drop_duplicates()
    log["duplicate_rows_removed"] = n - len(df)
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    n = len(df)
    df = df.dropna(subset=["CustomerID"])
    log["missing_customer_id_removed"] = n - len(df)
    df["CustomerID"] = df["CustomerID"].astype(float).astype(int)
    n = len(df)
    nonprod = df["StockCode"].str.upper().isin(NON_PRODUCT_CODES) | \
        df["StockCode"].str.lower().str.startswith("gift_")
    df = df[~nonprod]
    log["non_product_lines_removed"] = n - len(df)
    cancel = df["Invoice"].str.upper().str.startswith("C")
    returns = df[cancel & (df["Quantity"] < 0) & (df["Price"] > 0)].copy()
    sales = df[~cancel & (df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    log["cancellation_lines"] = len(returns)
    log["invalid_qty_or_price_removed"] = int(len(df) - len(returns) - len(sales))
    for d in (sales, returns):
        d["Revenue"] = d["Quantity"].abs() * d["Price"]
    log["valid_sales_lines"] = len(sales)
    return sales.reset_index(drop=True), returns.reset_index(drop=True), log


def default_cutoff(sales: pd.DataFrame, horizon: int = HORIZON_DAYS) -> pd.Timestamp:
    """Snapshot date such that the final `horizon` days of the data are the prediction window."""
    return sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(days=horizon)


def latest_snapshot(sales: pd.DataFrame) -> pd.Timestamp:
    """Snapshot just after the last transaction (used to score new data - no target exists)."""
    return sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)


# =========================================================================== 2. FEATURE ENGINEERING
def build_customer_table(sales, returns, cutoff, horizon: int = HORIZON_DAYS,
                         with_target: bool = True) -> pd.DataFrame:
    """One row per customer using ONLY data before `cutoff`; targets from [cutoff, cutoff+horizon)."""
    cutoff = pd.Timestamp(cutoff)
    hist = sales[sales["InvoiceDate"] < cutoff]
    if hist.empty:
        raise ValueError("No transactions found before the snapshot date.")
    inv = hist.groupby(["CustomerID", "Invoice"]).agg(
        InvoiceDate=("InvoiceDate", "max"), Value=("Revenue", "sum")).reset_index()
    ig = inv.groupby("CustomerID")
    cust = pd.DataFrame({"First": ig["InvoiceDate"].min(), "Last": ig["InvoiceDate"].max(),
                         "Frequency": ig.size(), "Monetary": ig["Value"].sum()})
    lg = hist.groupby("CustomerID").agg(TotalQuantity=("Quantity", "sum"),
                                        UniqueProducts=("StockCode", "nunique"),
                                        Lines=("StockCode", "size"))
    cust = cust.join(lg)
    cust["Country"] = hist.sort_values("InvoiceDate").groupby("CustomerID")["Country"].last()
    cust["Recency"] = (cutoff - cust["Last"]).dt.days
    cust["TenureDays"] = (cutoff - cust["First"]).dt.days
    span = (cust["Last"] - cust["First"]).dt.days
    cust["AvgDaysBetweenOrders"] = np.where(cust["Frequency"] > 1, span / (cust["Frequency"] - 1).clip(lower=1), np.nan)
    cust["AvgOrderValue"] = cust["Monetary"] / cust["Frequency"]
    cust["AvgLinesPerOrder"] = cust["Lines"] / cust["Frequency"]
    recent = inv[inv["InvoiceDate"] >= cutoff - pd.Timedelta(days=90)].groupby("CustomerID")
    cust["Orders_Last90"] = recent.size().reindex(cust.index).fillna(0).astype(int)
    cust["Spend_Last90"] = recent["Value"].sum().reindex(cust.index).fillna(0.0)
    ret = returns[returns["InvoiceDate"] < cutoff].groupby("CustomerID")["Revenue"].sum()
    cust["ReturnRate"] = (ret.reindex(cust.index).fillna(0.0) / cust["Monetary"]).clip(0, 1)
    cust["CountryGroup"] = np.where(cust["Country"].isin(TOP_COUNTRIES), cust["Country"], "Other")
    out = cust[FEATURES].copy()
    out["Country"] = cust["Country"]
    if with_target:
        end = cutoff + pd.Timedelta(days=horizon)
        fut = sales[(sales["InvoiceDate"] >= cutoff) & (sales["InvoiceDate"] < end)] \
            .groupby("CustomerID")["Revenue"].sum()
        out["FutureSpend"] = fut.reindex(out.index).fillna(0.0)
        out["Active"] = (out["FutureSpend"] > 0).astype(int)
    out = out.reset_index().rename(columns={"index": "CustomerID"})
    return out


# =========================================================================== 3. PREPROCESSING PIPELINES
def _log1p(X):
    """Module-level (picklable) log transform for right-skewed money/count features."""
    return np.log1p(np.clip(X, 0, None))


def make_preprocessor(scale: bool = True, log: bool = True) -> ColumnTransformer:
    """Same preprocessing is used in training AND inference because it lives inside the pipeline.
    impute (median) -> log1p -> standardise for numeric; one-hot for categorical."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if log:
        steps.append(("log", FunctionTransformer(_log1p, feature_names_out="one-to-one")))
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), NUM_FEATURES),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)])


def make_cluster_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                                ("log", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
                                                ("scale", StandardScaler())]), CLUSTER_FEATURES)])


def regression_models() -> dict:
    def wrap(est, scaled=True):
        pre = make_preprocessor(scale=scaled, log=scaled)
        return TransformedTargetRegressor(Pipeline([("prep", pre), ("model", est)]),
                                          func=np.log1p, inverse_func=np.expm1)
    simple = TransformedTargetRegressor(
        Pipeline([("prep", ColumnTransformer([("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler())]), ["Monetary"])])), ("model", LinearRegression())]),
        func=np.log1p, inverse_func=np.expm1)
    return {
        "Mean baseline": TransformedTargetRegressor(DummyRegressor(), func=np.log1p, inverse_func=np.expm1),
        "Simple Linear (Monetary)": simple,
        "Multiple Linear": wrap(LinearRegression()),
        "Ridge": wrap(Ridge(alpha=10.0)),
        "Random Forest": wrap(RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                                    random_state=RANDOM_STATE), scaled=False),
        "Hist Gradient Boosting": wrap(HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                                                     max_iter=250, random_state=RANDOM_STATE), scaled=False),
    }


def classification_models() -> dict:
    def pipe(est, scaled=True):
        return Pipeline([("prep", make_preprocessor(scale=scaled, log=scaled)), ("model", est)])
    return {
        "Logistic Regression": pipe(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "KNN (k=15)": pipe(KNeighborsClassifier(n_neighbors=15)),
        "Gaussian Naive Bayes": pipe(GaussianNB()),
        "Decision Tree": pipe(DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=RANDOM_STATE), False),
        "Bagging": pipe(BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=5), n_estimators=100,
                                          n_jobs=-1, random_state=RANDOM_STATE), False),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                                     random_state=RANDOM_STATE), False),
        "Gradient Boosting": pipe(GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                                             random_state=RANDOM_STATE), False),
    }


# =========================================================================== 4. SEGMENTATION
def fit_segmenter(cust: pd.DataFrame, k: int) -> dict:
    """K-Means on log-scaled RFM. Cluster ids are re-ordered so 0 = highest median spend."""
    pipe = Pipeline([("prep", make_cluster_preprocessor()),
                     ("km", KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE))])
    raw_labels = pipe.fit_predict(cust[CLUSTER_FEATURES])
    med = pd.Series(cust["Monetary"].values).groupby(raw_labels).median().sort_values(ascending=False)
    label_map = np.zeros(k, dtype=int)
    for new, old in enumerate(med.index):
        label_map[old] = new
    labels = label_map[raw_labels]
    profile = segment_profile(cust, labels)
    names = name_segments(profile, cust)
    return {"pipeline": pipe, "label_map": label_map, "names": names, "k": k, "features": CLUSTER_FEATURES}


def assign_segments(seg: dict, df: pd.DataFrame) -> np.ndarray:
    return seg["label_map"][seg["pipeline"].predict(df[seg["features"]])]


def segment_profile(cust: pd.DataFrame, labels) -> pd.DataFrame:
    d = cust.copy()
    d["Segment"] = np.asarray(labels)
    g = d.groupby("Segment")
    prof = pd.DataFrame({
        "Customers": g.size(),
        "Recency": g["Recency"].median(), "Frequency": g["Frequency"].median(),
        "Monetary": g["Monetary"].median(), "AvgOrderValue": g["AvgOrderValue"].median(),
        "TotalRevenue": g["Monetary"].sum()})
    prof["% Customers"] = 100 * prof["Customers"] / prof["Customers"].sum()
    prof["% Revenue"] = 100 * prof["TotalRevenue"] / prof["TotalRevenue"].sum()
    if "Active" in d.columns:
        prof["Repeat rate next 90d (%)"] = 100 * g["Active"].mean()
        prof["Avg future spend (GBP)"] = g["FutureSpend"].mean()
    return prof


def name_segments(profile: pd.DataFrame, cust: pd.DataFrame) -> dict:
    """Rule-based, data-driven names: each cluster median is compared to the overall median."""
    r0, f0, m0 = cust["Recency"].median(), cust["Frequency"].median(), cust["Monetary"].median()
    names, used = {}, {}
    for seg, row in profile.iterrows():
        r_hi, f_hi, m_hi = row["Recency"] > r0, row["Frequency"] > f0, row["Monetary"] > m0
        if m_hi and f_hi and not r_hi: nm = "Champions"
        elif m_hi and r_hi: nm = "At-Risk Big Spenders"
        elif m_hi: nm = "Big Basket Buyers"
        elif f_hi and not r_hi: nm = "Loyal Regulars"
        elif not r_hi: nm = "New & Promising"
        else: nm = "Lapsed Low-Value"
        used[nm] = used.get(nm, 0) + 1
        names[seg] = nm if used[nm] == 1 else f"{nm} ({chr(64 + used[nm])})"
    # first duplicate keeps plain name; label it (A) too for symmetry
    for nm, c in used.items():
        if c > 1:
            for s, v in names.items():
                if v == nm: names[s] = f"{nm} (A)"
    return names


ACTIONS = {
    "Champions": "Reward and retain: early access, loyalty perks, referral incentives. Protect margin - do not discount.",
    "At-Risk Big Spenders": "Win back fast: personalised outreach, 'we miss you' offer, reminders of favourite products.",
    "Big Basket Buyers": "Grow frequency: bundles, replenishment reminders, minimum-spend incentives for the next order.",
    "Loyal Regulars": "Increase basket size: cross-sell related products and volume tiers.",
    "New & Promising": "Nurture: onboarding emails and a second-order incentive to build the habit.",
    "Lapsed Low-Value": "Low-cost reactivation only (email), otherwise deprioritise marketing spend.",
}


def action_for(name: str) -> str:
    base = name.split(" (")[0]
    return ACTIONS.get(base, "Review this segment's profile and tailor the offer.")


def cluster_features_matrix(cust: pd.DataFrame) -> np.ndarray:
    return make_cluster_preprocessor().fit_transform(cust[CLUSTER_FEATURES])


def run_clustering(Z: np.ndarray, algo: str, k: int, **params):
    """Fit a clustering algorithm on scaled features. Returns (hard_labels, soft_probs_or_None)."""
    if algo == "K-Means":
        return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z), None
    if algo == "Gaussian Mixture (GMM)":
        gm = GaussianMixture(n_components=k, covariance_type=params.get("covariance_type", "full"),
                             random_state=RANDOM_STATE, n_init=3).fit(Z)
        return gm.predict(Z), gm.predict_proba(Z)
    if algo == "Agglomerative":
        return AgglomerativeClustering(n_clusters=k, linkage=params.get("linkage", "ward")).fit_predict(Z), None
    raise ValueError(f"Unknown algorithm {algo}")


def order_labels_by_spend(labels, cust: pd.DataFrame) -> np.ndarray:
    labels = np.asarray(labels)
    med = pd.Series(cust["Monetary"].values).groupby(labels).median().sort_values(ascending=False)
    m = {old: new for new, old in enumerate(med.index)}
    return np.array([m[l] for l in labels])


# =========================================================================== 5. METRICS HELPERS
def reg_metrics(y_true, y_pred) -> dict:
    y_pred = np.clip(y_pred, 0, None)
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred),
            "R2 (log scale)": r2_score(np.log1p(y_true), np.log1p(y_pred))}


def clf_metrics(y_true, y_pred, y_prob) -> dict:
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_prob)}


# =========================================================================== 6. AGGREGATES FOR THE DASHBOARD
def build_aggregates(raw: pd.DataFrame, sales: pd.DataFrame, returns: pd.DataFrame, log: dict) -> dict:
    kpis = {"total_revenue": float(sales["Revenue"].sum()), "return_value": float(returns["Revenue"].sum()),
            "n_invoices": int(sales["Invoice"].nunique()), "n_customers": int(sales["CustomerID"].nunique()),
            "n_products": int(sales["StockCode"].nunique()), "n_countries": int(sales["Country"].nunique()),
            "date_min": sales["InvoiceDate"].min(), "date_max": sales["InvoiceDate"].max()}
    kpis["aov"] = kpis["total_revenue"] / kpis["n_invoices"]
    m = sales.assign(Month=sales["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
    monthly = m.groupby("Month").agg(Revenue=("Revenue", "sum"), Orders=("Invoice", "nunique"),
                                     Customers=("CustomerID", "nunique")).reset_index()
    country = sales.groupby("Country").agg(Revenue=("Revenue", "sum"), Customers=("CustomerID", "nunique"),
                                           Orders=("Invoice", "nunique")).sort_values("Revenue", ascending=False).reset_index()
    prods = sales.groupby("StockCode").agg(Description=("Description", "last"), Revenue=("Revenue", "sum"),
                                           Units=("Quantity", "sum")).sort_values("Revenue", ascending=False).head(15).reset_index()
    heat = sales.assign(Weekday=sales["InvoiceDate"].dt.day_name(), Hour=sales["InvoiceDate"].dt.hour) \
        .groupby(["Weekday", "Hour"])["Revenue"].sum().reset_index()
    prof = pd.DataFrame({"Column": raw.columns, "Type": [str(t) for t in raw.dtypes],
                         "Missing": raw.isna().sum().values,
                         "Missing %": (100 * raw.isna().mean()).round(2).values,
                         "Unique": raw.nunique().values,
                         "Example": [str(raw[c].dropna().iloc[0]) if raw[c].notna().any() else "" for c in raw.columns]})
    sample = sales[["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"]].head(1000)
    return {"kpis": kpis, "monthly": monthly, "country": country, "top_products": prods, "heat": heat,
            "raw_profile": prof, "raw_describe": raw[["Quantity", "Price"]].describe().T.reset_index(),
            "sample": sample, "cleaning_log": log}


# =========================================================================== 7. TRAINING
def train_all(data_path=DEFAULT_DATA_FILE, progress=print) -> dict:
    """Reproducible end-to-end training. Writes models/*.joblib and data/processed/customers.csv."""
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    PROC_DIR.mkdir(exist_ok=True, parents=True)
    progress("Loading dataset ...")
    raw = load_raw_cached(data_path)
    progress("Cleaning transactions ...")
    sales, returns, log = clean_transactions(raw)
    cutoff = default_cutoff(sales)
    cust = build_customer_table(sales, returns, cutoff)
    cust.to_csv(PROC_DIR / "customers.csv", index=False)
    agg = build_aggregates(raw, sales, returns, log)
    progress(f"{len(cust):,} customers as of {cutoff.date()}; repeat rate in next {HORIZON_DAYS}d = {cust['Active'].mean():.1%}")

    X, y_reg, y_clf = cust[FEATURES], cust["FutureSpend"], cust["Active"]
    Xtr, Xte, ytr_r, yte_r, ytr_c, yte_c = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf)
    ids_te = cust.loc[Xte.index, "CustomerID"].values

    # ---------------- regression
    progress("Training regression models ...")
    kf = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    reg_rows, reg_fitted, reg_preds = [], {}, {}
    for name, mdl in regression_models().items():
        cv = cross_val_score(mdl, Xtr, ytr_r, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=1)
        mdl.fit(Xtr, ytr_r)
        pred = np.clip(mdl.predict(Xte), 0, None)
        reg_rows.append({"Model": name, **reg_metrics(yte_r, pred), "CV RMSE (train)": -cv.mean()})
        reg_fitted[name], reg_preds[name] = mdl, pred
    reg_tbl = pd.DataFrame(reg_rows).set_index("Model")
    cand = reg_tbl.drop(index=["Mean baseline"])
    best_reg = cand["RMSE"].idxmin()
    progress(f"Best regression model: {best_reg}")
    pi = permutation_importance(reg_fitted[best_reg], Xte, yte_r, n_repeats=5, random_state=RANDOM_STATE,
                                scoring="neg_root_mean_squared_error", n_jobs=1)
    reg_imp = pd.DataFrame({"Feature": FEATURES, "Importance": pi.importances_mean}).sort_values("Importance", ascending=False)

    # ---------------- classification
    progress("Training classification models ...")
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    clf_rows, clf_fitted, clf_cm, clf_roc, clf_prob = [], {}, {}, {}, {}
    for name, mdl in classification_models().items():
        cv = cross_val_score(mdl, Xtr, ytr_c, cv=skf, scoring="roc_auc", n_jobs=1)
        mdl.fit(Xtr, ytr_c)
        prob = mdl.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        clf_rows.append({"Model": name, **clf_metrics(yte_c, pred, prob), "CV AUC mean": cv.mean(), "CV AUC std": cv.std()})
        clf_fitted[name] = mdl
        clf_cm[name] = confusion_matrix(yte_c, pred)
        fpr, tpr, _ = roc_curve(yte_c, prob)
        clf_roc[name] = (fpr, tpr)
        clf_prob[name] = prob
    # tuned random forest (RandomizedSearchCV) as an extra candidate
    progress("Tuning Random Forest with RandomizedSearchCV ...")
    rs = RandomizedSearchCV(classification_models()["Random Forest"],
                            {"model__n_estimators": [200, 300, 500], "model__max_depth": [4, 6, 8, 12, None],
                             "model__min_samples_leaf": [2, 5, 10, 20], "model__max_features": ["sqrt", 0.5, 0.8]},
                            n_iter=12, cv=3, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=1)
    rs.fit(Xtr, ytr_c)
    name = "Random Forest (tuned)"
    mdl = rs.best_estimator_
    prob = mdl.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    clf_rows.append({"Model": name, **clf_metrics(yte_c, pred, prob), "CV AUC mean": rs.best_score_, "CV AUC std": np.nan})
    clf_fitted[name], clf_cm[name] = mdl, confusion_matrix(yte_c, pred)
    fpr, tpr, _ = roc_curve(yte_c, prob)
    clf_roc[name], clf_prob[name] = (fpr, tpr), prob
    clf_tbl = pd.DataFrame(clf_rows).set_index("Model")
    best_clf = clf_tbl["CV AUC mean"].idxmax()
    progress(f"Best classifier (by CV AUC): {best_clf}")
    pi = permutation_importance(clf_fitted[best_clf], Xte, yte_c, n_repeats=5, random_state=RANDOM_STATE,
                                scoring="roc_auc", n_jobs=1)
    clf_imp = pd.DataFrame({"Feature": FEATURES, "Importance": pi.importances_mean}).sort_values("Importance", ascending=False)

    # ---------------- clustering
    progress("Fitting customer segments ...")
    Z = cluster_features_matrix(cust)
    ks = []
    for k in range(2, 9):
        lab = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        ks.append({"k": k, "Inertia": lab.inertia_, "Silhouette": silhouette_score(Z, lab.labels_, sample_size=4000, random_state=RANDOM_STATE)})
    ks = pd.DataFrame(ks)
    best_k = int(ks[(ks.k >= 3) & (ks.k <= 6)].sort_values("Silhouette", ascending=False).iloc[0]["k"])
    seg = fit_segmenter(cust, best_k)
    cust["Segment"] = assign_segments(seg, cust)
    prof = segment_profile(cust, cust["Segment"].values)
    prof.insert(0, "Name", [seg["names"][i] for i in prof.index])

    meta = {"cutoff": cutoff, "horizon_days": HORIZON_DAYS, "n_customers": len(cust), "n_train": len(Xtr), "n_test": len(Xte),
            "active_rate": float(cust["Active"].mean()), "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__, "features": FEATURES}
    results = {"meta": meta,
               "regression": {"table": reg_tbl, "best": best_reg, "importance": reg_imp,
                              "test": pd.DataFrame({"CustomerID": ids_te, "Actual": yte_r.values, "Predicted": reg_preds[best_reg]}),
                              "all_preds": reg_preds},
               "classification": {"table": clf_tbl, "best": best_clf, "importance": clf_imp, "confusion": clf_cm,
                                  "roc": clf_roc, "best_params": rs.best_params_},
               "clustering": {"k_scores": ks, "k": best_k, "profile": prof}}
    joblib.dump({"models": {"regression": reg_fitted[best_reg], "classification": clf_fitted[best_clf]},
                 "names": {"regression": best_reg, "classification": best_clf}, "meta": meta},
                MODEL_DIR / "models.joblib", compress=3)
    joblib.dump(seg, MODEL_DIR / "segmenter.joblib", compress=3)
    joblib.dump(results, MODEL_DIR / "results.joblib", compress=3)
    joblib.dump(agg, MODEL_DIR / "aggregates.joblib", compress=3)
    cust.to_csv(PROC_DIR / "customers.csv", index=False)
    progress("Done - artifacts saved to models/")
    return results


# =========================================================================== 8. INFERENCE
def artifacts_exist() -> bool:
    return all((MODEL_DIR / f).exists() for f in ["models.joblib", "segmenter.joblib", "results.joblib", "aggregates.joblib"]) \
        and (PROC_DIR / "customers.csv").exists()


def load_artifacts() -> dict:
    a = {"models": joblib.load(MODEL_DIR / "models.joblib"), "segmenter": joblib.load(MODEL_DIR / "segmenter.joblib"),
         "results": joblib.load(MODEL_DIR / "results.joblib"), "agg": joblib.load(MODEL_DIR / "aggregates.joblib"),
         "customers": pd.read_csv(PROC_DIR / "customers.csv")}
    return a


def validate_inputs(d: dict) -> list[str]:
    errs = []
    for k in ["Recency", "Frequency", "Monetary", "TotalQuantity", "UniqueProducts", "AvgLinesPerOrder",
              "TenureDays", "Orders_Last90", "Spend_Last90", "ReturnRate"]:
        v = d.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errs.append(f"{k} is required."); continue
        if v < 0: errs.append(f"{k} cannot be negative.")
    if not errs:
        if d["Frequency"] < 1: errs.append("Frequency must be at least 1 order.")
        if d["Recency"] > d["TenureDays"]: errs.append("Recency cannot exceed Tenure (last purchase must be after first purchase).")
        if d["ReturnRate"] > 1: errs.append("ReturnRate must be between 0 and 1.")
        if d["Spend_Last90"] > d["Monetary"] + 1e-6: errs.append("Spend in the last 90 days cannot exceed total spend.")
        if d["UniqueProducts"] > d["TotalQuantity"]: errs.append("Unique products cannot exceed total quantity.")
    return errs


def to_frame(d: dict) -> pd.DataFrame:
    d = dict(d)
    d["AvgOrderValue"] = d["Monetary"] / max(d["Frequency"], 1)
    if d.get("AvgDaysBetweenOrders") is None:
        d["AvgDaysBetweenOrders"] = np.nan
    if d["CountryGroup"] not in TOP_COUNTRIES + ["Other"]:
        d["CountryGroup"] = "Other"
    return pd.DataFrame([{c: d.get(c, np.nan) for c in FEATURES}])


def predict_one(art: dict, d: dict) -> dict:
    X = to_frame(d)
    reg, clf = art["models"]["models"]["regression"], art["models"]["models"]["classification"]
    spend = float(np.clip(reg.predict(X)[0], 0, None))
    proba = clf.predict_proba(X)[0]
    seg = int(assign_segments(art["segmenter"], X)[0])
    return {"spend": spend, "proba": proba, "prob_active": float(proba[1]), "segment": seg,
            "segment_name": art["segmenter"]["names"].get(seg, str(seg))}


def score_transactions(art: dict, raw: pd.DataFrame) -> pd.DataFrame:
    """Score a user-uploaded transaction file: build customer features at the latest snapshot,
    assign segments, predict next-90-day spend and repeat probability."""
    sales, returns, _ = clean_transactions(raw)
    if sales.empty:
        raise ValueError("No valid purchase rows after cleaning (check Quantity/Price/CustomerID).")
    c = build_customer_table(sales, returns, latest_snapshot(sales), with_target=False)
    c["Segment"] = assign_segments(art["segmenter"], c)
    c["SegmentName"] = c["Segment"].map(art["segmenter"]["names"])
    c["PredictedSpend90d"] = np.clip(art["models"]["models"]["regression"].predict(c[FEATURES]), 0, None)
    c["ProbRepeat90d"] = art["models"]["models"]["classification"].predict_proba(c[FEATURES])[:, 1]
    return c


# =========================================================================== 9. COMMAND LINE
if __name__ == "__main__":
    # python backend.py [path/to/dataset.xlsx|csv]  -> trains everything and writes models/ + data/processed/
    import sys
    train_all(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_FILE)
