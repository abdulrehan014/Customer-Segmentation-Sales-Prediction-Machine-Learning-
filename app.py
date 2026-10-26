"""Customer Segmentation & Sales Prediction - Streamlit dashboard.
Run:  streamlit run app.py      (models are loaded from models/, never retrained on rerun)"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import backend as B

st.set_page_config(page_title="Customer Segmentation & Sales Prediction", page_icon="📊", layout="wide")

PALETTE = ["#2E5BFF", "#00A98F", "#F5A623", "#E0457B", "#8B5CF6", "#64748B", "#14B8A6", "#F97316"]
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1300px;}
[data-testid="stMetric"] {background: var(--secondary-background-color); border: 1px solid rgba(128,128,128,.18);
    border-radius: 12px; padding: 14px 16px;}
[data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .75;}
h1, h2, h3 {letter-spacing: -0.01em;}
.note {border-left: 4px solid #2E5BFF; padding: .6rem 1rem; background: rgba(46,91,255,.07); border-radius: 6px; margin: .4rem 0 1rem 0;}
</style>""", unsafe_allow_html=True)


def fig_style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=45, b=10), legend_title_text="",
                      font=dict(size=13), colorway=PALETTE)
    return fig


def gbp(x):
    return f"£{x:,.0f}"


# ------------------------------------------------------------------ load artifacts (cached)
@st.cache_resource(show_spinner="Loading trained models ...")
def get_art():
    return B.load_artifacts()


@st.cache_data(show_spinner=False)
def cached_cluster_matrix(df: pd.DataFrame):
    return B.cluster_features_matrix(df)


@st.cache_data(show_spinner="Clustering customers ...")
def cached_clustering(algo, k, cov, linkage, _Z, _cust):
    labels, probs = B.run_clustering(_Z, algo, k, covariance_type=cov, linkage=linkage)
    labels = B.order_labels_by_spend(labels, _cust)
    sil = silhouette_score(_Z, labels, sample_size=min(4000, len(_Z)), random_state=B.RANDOM_STATE) if len(set(labels)) > 1 else float("nan")
    pca = PCA(2, random_state=B.RANDOM_STATE).fit(_Z)
    return labels, probs, sil, pca.transform(_Z), pca.explained_variance_ratio_


if not B.artifacts_exist():
    st.title("Customer Segmentation & Sales Prediction")
    st.warning("Trained model artifacts were not found in `models/`.")
    if B.DEFAULT_DATA_FILE.exists():
        st.write(f"Dataset found at `{B.DEFAULT_DATA_FILE.name}`. Train the models once (about 2-3 minutes):")
        if st.button("Train models now", type="primary"):
            with st.spinner("Training ... this runs once and the artifacts are saved to models/"):
                B.train_all()
            st.success("Done - reloading.")
            st.rerun()
    else:
        st.write("Place the dataset at `data/online_retail_II.xlsx` and run `python backend.py`.")
    st.stop()

try:
    art = get_art()
except Exception as e:  # version mismatch, corrupt files ...
    st.error(f"Could not load model artifacts: {e}")
    st.info("Re-create them for this environment with `python backend.py`.")
    st.stop()

res, agg, meta, seg = art["results"], art["agg"], art["results"]["meta"], art["segmenter"]
cust = art["customers"].copy()
cust["SegmentName"] = cust["Segment"].map(seg["names"])
seg_order = [seg["names"][i] for i in sorted(seg["names"])]
seg_colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(seg_order)}
import sklearn
if meta["sklearn_version"].split(".")[:2] != sklearn.__version__.split(".")[:2]:
    st.sidebar.warning(f"Models were trained with scikit-learn {meta['sklearn_version']} but this environment has "
                       f"{sklearn.__version__}. If anything misbehaves, run `python backend.py`.")

# ------------------------------------------------------------------ sidebar
st.sidebar.title("📊 Customer Analytics")
page = st.sidebar.radio("Navigate", ["Dashboard Overview", "Data Explorer", "Customer Segmentation", "Sales Prediction",
                                     "Classification", "Model Comparison", "Model Insights"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(f"Snapshot date: **{meta['cutoff'].date()}**  \nPrediction window: next **{meta['horizon_days']} days**  \n"
                   f"Customers modelled: **{meta['n_customers']:,}**")
st.sidebar.caption("Dataset: UCI Online Retail II (2009-2011)")


# ------------------------------------------------------------------ shared customer input form
def customer_form(key: str):
    """Inputs for the real model features. Returns (dict, valid)."""
    mode = st.radio("Input mode", ["Existing customer from the dataset", "Enter values manually"], horizontal=True, key=f"mode_{key}")
    q = cust[B.NUM_FEATURES]
    if mode.startswith("Existing"):
        cid = st.selectbox("Customer ID", cust["CustomerID"].tolist(), key=f"cid_{key}")
        row = cust[cust["CustomerID"] == cid].iloc[0]
        vals = {f: (None if pd.isna(row[f]) else float(row[f])) for f in B.NUM_FEATURES}
        vals["CountryGroup"] = row["CountryGroup"]
        st.caption("Features shown are the customer's real values at the snapshot date. "
                   f"Actual spend in the following {meta['horizon_days']} days was **{gbp(row['FutureSpend'])}** "
                   f"({'purchased' if row['Active'] else 'did not purchase'}).")
        st.dataframe(pd.DataFrame([vals]).T.rename(columns={0: "Value"}).T, width="stretch", hide_index=True)
        return vals, True, row
    med = q.median()
    c1, c2, c3 = st.columns(3)
    d = {}
    with c1:
        d["Recency"] = st.number_input("Recency (days since last order)", 0.0, 800.0, float(med["Recency"]), 1.0, key=f"r{key}", help=B.FEATURE_HELP["Recency"])
        d["Frequency"] = st.number_input("Frequency (orders)", 1.0, 500.0, float(max(med["Frequency"], 1)), 1.0, key=f"f{key}", help=B.FEATURE_HELP["Frequency"])
        d["Monetary"] = st.number_input("Monetary (total spend £)", 0.0, 1_000_000.0, float(round(med["Monetary"])), 50.0, key=f"m{key}", help=B.FEATURE_HELP["Monetary"])
        d["TenureDays"] = st.number_input("Tenure (days since first order)", 0.0, 800.0, float(med["TenureDays"]), 1.0, key=f"t{key}", help=B.FEATURE_HELP["TenureDays"])
    with c2:
        d["TotalQuantity"] = st.number_input("Total units bought", 0.0, 500_000.0, float(med["TotalQuantity"]), 10.0, key=f"q{key}", help=B.FEATURE_HELP["TotalQuantity"])
        d["UniqueProducts"] = st.number_input("Distinct products", 0.0, 5000.0, float(med["UniqueProducts"]), 1.0, key=f"u{key}", help=B.FEATURE_HELP["UniqueProducts"])
        d["AvgLinesPerOrder"] = st.number_input("Avg product lines per order", 0.0, 500.0, float(round(med["AvgLinesPerOrder"], 1)), 0.5, key=f"l{key}", help=B.FEATURE_HELP["AvgLinesPerOrder"])
        d["ReturnRate"] = st.slider("Return rate (share of spend)", 0.0, 1.0, float(round(med["ReturnRate"], 2)), 0.01, key=f"rr{key}", help=B.FEATURE_HELP["ReturnRate"])
    with c3:
        d["Orders_Last90"] = st.number_input("Orders in the last 90 days", 0.0, 200.0, float(med["Orders_Last90"]), 1.0, key=f"o{key}", help=B.FEATURE_HELP["Orders_Last90"])
        d["Spend_Last90"] = st.number_input("Spend in the last 90 days (£)", 0.0, 1_000_000.0, float(round(med["Spend_Last90"])), 50.0, key=f"s{key}", help=B.FEATURE_HELP["Spend_Last90"])
        known = st.checkbox("Average days between orders is known", value=d["Frequency"] > 1, key=f"k{key}")
        d["AvgDaysBetweenOrders"] = st.number_input("Avg days between orders", 0.0, 800.0, float(med["AvgDaysBetweenOrders"]), 1.0, key=f"g{key}", disabled=not known) if known else None
        d["CountryGroup"] = st.selectbox("Country", B.TOP_COUNTRIES + ["Other"], key=f"c{key}")
    errs = B.validate_inputs(d)
    for e in errs:
        st.error(e)
    return d, not errs, None


# =================================================================== PAGES
if page == "Dashboard Overview":
    st.title("Dashboard Overview")
    k = agg["kpis"]
    st.caption(f"Cleaned transactions from {k['date_min']:%d %b %Y} to {k['date_max']:%d %b %Y}")
    c = st.columns(5)
    c[0].metric("Total revenue", f"£{k['total_revenue']/1e6:,.2f}M")
    c[1].metric("Customers", f"{k['n_customers']:,}")
    c[2].metric("Orders (invoices)", f"{k['n_invoices']:,}")
    c[3].metric("Avg order value", gbp(k["aov"]))
    c[4].metric("Returned value", f"£{k['return_value']/1e3:,.0f}K", f"{100*k['return_value']/k['total_revenue']:.1f}% of revenue", delta_color="off")
    c = st.columns(3)
    c[0].metric("Products sold", f"{k['n_products']:,}")
    c[1].metric("Countries", k["n_countries"])
    c[2].metric(f"Customers who re-purchased in next {meta['horizon_days']}d", f"{100*meta['active_rate']:.1f}%")

    m = agg["monthly"]
    l, r = st.columns([3, 2])
    fig = px.area(m, x="Month", y="Revenue", title="Monthly revenue (GBP)")
    l.plotly_chart(fig_style(fig), width="stretch")
    sc = cust["SegmentName"].value_counts().reindex(seg_order)
    fig = px.pie(values=sc.values, names=sc.index, hole=0.55, title="Customers by segment", color=sc.index, color_discrete_map=seg_colors)
    r.plotly_chart(fig_style(fig), width="stretch")
    l, r = st.columns(2)
    top = agg["country"].head(8)
    l.plotly_chart(fig_style(px.bar(top, x="Revenue", y="Country", orientation="h", title="Revenue by country (top 8)").update_yaxes(autorange="reversed")), width="stretch")
    rev = cust.groupby("SegmentName")["Monetary"].sum().reindex(seg_order)
    r.plotly_chart(fig_style(px.bar(x=rev.index, y=rev.values, color=rev.index, color_discrete_map=seg_colors,
                                    title="Historic revenue by segment (GBP)", labels={"x": "", "y": "Revenue"}).update_layout(showlegend=False)), width="stretch")

elif page == "Data Explorer":
    st.title("Data Explorer")
    t1, t2, t3, t4 = st.tabs(["Transactions preview", "Columns & missing values", "Customer feature table", "Interactive charts"])
    with t1:
        st.caption("First 1,000 rows of the cleaned transaction data (valid purchases).")
        st.dataframe(agg["sample"], width="stretch", height=380)
        st.subheader("Cleaning log")
        log = pd.DataFrame({"Step": [k.replace("_", " ").capitalize() for k in agg["cleaning_log"]], "Rows": list(agg["cleaning_log"].values())})
        st.dataframe(log, width="stretch", hide_index=True)
    with t2:
        st.subheader("Raw dataset schema")
        st.dataframe(agg["raw_profile"], width="stretch", hide_index=True)
        st.caption("Customer ID is missing on ~22% of rows (guest checkouts) - those rows cannot be attributed to a customer and are excluded from customer-level modelling.")
        st.subheader("Numeric summary (raw)")
        st.dataframe(agg["raw_describe"], width="stretch", hide_index=True)
    with t3:
        st.caption("One row per customer, features computed strictly before the snapshot date. Targets: FutureSpend, Active.")
        f1, f2 = st.columns(2)
        seg_f = f1.multiselect("Segment", seg_order, default=seg_order)
        ctry = f2.multiselect("Country group", B.TOP_COUNTRIES + ["Other"], default=B.TOP_COUNTRIES + ["Other"])
        view = cust[cust["SegmentName"].isin(seg_f) & cust["CountryGroup"].isin(ctry)]
        st.write(f"{len(view):,} customers")
        st.dataframe(view.drop(columns=["Segment"]), width="stretch", height=330)
        st.download_button("Download filtered table (CSV)", view.to_csv(index=False).encode(), "customers_filtered.csv", "text/csv")
        st.subheader("Summary statistics")
        st.dataframe(view[B.NUM_FEATURES + ["FutureSpend"]].describe().T.round(2), width="stretch")
        st.subheader("Missing values")
        miss = view[B.FEATURES].isna().sum()
        st.dataframe(miss[miss > 0].rename("Missing").to_frame().T if miss.sum() else pd.DataFrame({"Result": ["No missing values"]}), width="stretch", hide_index=True)
        st.caption("AvgDaysBetweenOrders is undefined for one-order customers; it is median-imputed inside the model pipelines.")
    with t4:
        view = cust
        kind = st.radio("Chart", ["Histogram", "Box plot", "Scatter", "Correlation heatmap"], horizontal=True)
        num = B.NUM_FEATURES + ["FutureSpend"]
        by = st.checkbox("Colour by segment", True)
        if kind == "Histogram":
            col = st.selectbox("Feature", num, index=2)
            logx = st.checkbox("Log x-axis", True)
            fig = px.histogram(view[view[col] > 0] if logx else view, x=col, color="SegmentName" if by else None, nbins=60,
                               log_x=logx, color_discrete_map=seg_colors, category_orders={"SegmentName": seg_order})
        elif kind == "Box plot":
            col = st.selectbox("Feature", num, index=2)
            fig = px.box(view, x="SegmentName", y=col, color="SegmentName", log_y=st.checkbox("Log y-axis", True) and (view[col] > 0).all(),
                         color_discrete_map=seg_colors, category_orders={"SegmentName": seg_order})
        elif kind == "Scatter":
            a, b = st.columns(2)
            x = a.selectbox("X", num, index=2)
            y = b.selectbox("Y", num, index=len(num) - 1)
            fig = px.scatter(view, x=x, y=y, color="SegmentName" if by else None, opacity=.55, log_x=True, log_y=True,
                             color_discrete_map=seg_colors, category_orders={"SegmentName": seg_order}, hover_data=["CustomerID"])
        else:
            fig = px.imshow(view[num].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig_style(fig, 480), width="stretch")

elif page == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.markdown('<div class="note">Segments are built from <b>RFM</b> (Recency, Frequency, Monetary) after a log transform and standardisation. '
                'No target information is used, so segments are valid for new customers too.</div>', unsafe_allow_html=True)
    c = st.columns(4)
    algo = c[0].selectbox("Algorithm", ["K-Means", "Gaussian Mixture (GMM)", "Agglomerative"])
    k = c[1].slider("Number of clusters", 2, 8, seg["k"])
    cov = c[2].selectbox("GMM covariance", ["full", "tied", "diag", "spherical"], disabled=algo != "Gaussian Mixture (GMM)")
    link = c[3].selectbox("Linkage", ["ward", "complete", "average"], disabled=algo != "Agglomerative")
    Z = cached_cluster_matrix(cust)
    labels, probs, sil, P, var = cached_clustering(algo, k, cov, link, Z, cust)
    d = cust.copy()
    d["Cluster"] = labels
    prof = B.segment_profile(d, labels)
    use_named = algo == "K-Means" and k == seg["k"]
    names = seg["names"] if use_named else B.name_segments(prof, d)
    d["Label"] = d["Cluster"].map(names)
    order = [names[i] for i in sorted(names)]
    cmap = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(order)}
    m = st.columns(3)
    m[0].metric("Silhouette score", f"{sil:.3f}", help="Higher = better separated (-1 to 1)")
    m[1].metric("Clusters", k)
    m[2].metric("PCA variance shown", f"{100*var.sum():.0f}%")
    l, r = st.columns([3, 2])
    pdf = pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1], "Segment": d["Label"], "CustomerID": d["CustomerID"]})
    fig = px.scatter(pdf, x="PC1", y="PC2", color="Segment", opacity=.6, hover_data=["CustomerID"], color_discrete_map=cmap,
                     category_orders={"Segment": order}, title="Customer clusters (PCA projection of scaled RFM)", render_mode="webgl")
    l.plotly_chart(fig_style(fig, 440), width="stretch")
    sizes = d["Label"].value_counts().reindex(order)
    r.plotly_chart(fig_style(px.bar(x=sizes.values, y=sizes.index, orientation="h", color=sizes.index, color_discrete_map=cmap,
                                    title="Cluster sizes", labels={"x": "Customers", "y": ""}).update_layout(showlegend=False).update_yaxes(autorange="reversed"), 440), width="stretch")
    st.subheader("Cluster profiles (medians)")
    show = prof.copy()
    show.insert(0, "Segment", [names[i] for i in show.index])
    show = show.drop(columns=["TotalRevenue"]).round(1)
    st.dataframe(show, width="stretch", hide_index=True)
    zs = d.groupby("Label")[B.CLUSTER_FEATURES].median().reindex(order)
    zs = (np.log1p(zs) - np.log1p(cust[B.CLUSTER_FEATURES]).median()) / np.log1p(cust[B.CLUSTER_FEATURES]).std()
    st.plotly_chart(fig_style(px.imshow(zs.round(2), text_auto=True, color_continuous_scale="RdBu_r", color_continuous_midpoint=0, aspect="auto",
                                        title="Relative level of each RFM feature vs. all customers (log z-score; Recency: higher = more dormant)"), 330), width="stretch")
    if probs is not None:
        st.subheader("GMM soft membership")
        st.caption("A GMM gives every customer a probability of belonging to each cluster.")
        cid = st.selectbox("Customer", d["CustomerID"].tolist(), key="gmm_cid")
        idx = int(np.where(d["CustomerID"].values == cid)[0][0])
        order_idx = sorted(set(d["Cluster"]))
        # probs columns follow raw GMM component ids; remap to the spend-ordered cluster ids via majority assignment
        raw_hard = probs.argmax(1)
        mapping = {c_: d["Cluster"][raw_hard == c_].mode().iat[0] for c_ in range(probs.shape[1]) if (raw_hard == c_).any()}
        pr = pd.Series({names[mapping[c_]]: probs[idx, c_] for c_ in mapping}).groupby(level=0).sum().reindex(order).fillna(0)
        st.plotly_chart(fig_style(px.bar(x=pr.index, y=pr.values, color=pr.index, color_discrete_map=cmap, labels={"x": "", "y": "Probability"},
                                         title=f"Membership probabilities - customer {cid}").update_layout(showlegend=False), 300), width="stretch")
    st.subheader("Assign a segment to a new customer (saved K-Means model)")
    cc = st.columns(3)
    r_ = cc[0].number_input("Recency (days)", 0, 800, 30)
    f_ = cc[1].number_input("Frequency (orders)", 1, 500, 3)
    m_ = cc[2].number_input("Monetary (£)", 0.0, 1_000_000.0, 800.0, 50.0)
    sid = int(B.assign_segments(seg, pd.DataFrame([{"Recency": r_, "Frequency": f_, "Monetary": m_}]))[0])
    st.success(f"Segment: **{seg['names'][sid]}**")
    st.caption(B.action_for(seg["names"][sid]))
    with st.expander("Batch-score your own transaction file (CSV / Excel)"):
        up = st.file_uploader("Upload transactions (columns: Invoice, StockCode, Quantity, InvoiceDate, Price, Customer ID, Country)", type=["csv", "xlsx"], key="up_seg")
        if up is not None:
            try:
                with st.spinner("Scoring ..."):
                    out = B.score_transactions(art, B.load_raw(up, up.name))
                st.dataframe(out, width="stretch", height=300)
                st.download_button("Download scored customers", out.to_csv(index=False).encode(), "scored_customers.csv", "text/csv")
            except Exception as e:
                st.error(f"Could not process this file: {e}")

elif page == "Sales Prediction":
    st.title("Sales Prediction")
    st.markdown(f'<div class="note">Predicts how much a customer will spend in the <b>next {meta["horizon_days"]} days</b> (GBP). '
                f'Model: <b>{art["models"]["names"]["regression"]}</b>, trained on log(1 + spend) and converted back to GBP. '
                f'Test RMSE {gbp(res["regression"]["table"].loc[res["regression"]["best"], "RMSE"])}, '
                f'MAE {gbp(res["regression"]["table"].loc[res["regression"]["best"], "MAE"])}.</div>', unsafe_allow_html=True)
    vals, ok, row = customer_form("reg")
    if st.button("Predict next-90-day spend", type="primary", disabled=not ok):
        try:
            with st.spinner("Predicting ..."):
                out = B.predict_one(art, vals)
            a, b, c = st.columns(3)
            a.metric("Predicted spend", gbp(out["spend"]))
            b.metric("Segment", out["segment_name"])
            if row is not None:
                c.metric("Actual spend (holdout)", gbp(row["FutureSpend"]), f"{out['spend']-row['FutureSpend']:+,.0f} vs predicted", delta_color="off")
            st.caption("Spend is highly skewed: expect large relative error for individual customers; the model is most useful for ranking and totals.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    st.divider()
    st.subheader("Held-out test performance")
    t = res["regression"]["test"]
    l, r = st.columns(2)
    fig = px.scatter(t, x="Actual", y="Predicted", opacity=.5, log_x=True, log_y=True, hover_data=["CustomerID"], title="Actual vs predicted (log axes, test set)")
    mx = max(t.Actual.max(), t.Predicted.max())
    fig.add_trace(go.Scatter(x=[1, mx], y=[1, mx], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect"))
    l.plotly_chart(fig_style(fig), width="stretch")
    r.plotly_chart(fig_style(px.bar(res["regression"]["importance"].head(10), x="Importance", y="Feature", orientation="h",
                                    title="Permutation importance (RMSE increase, GBP)").update_yaxes(autorange="reversed")), width="stretch")

elif page == "Classification":
    st.title("Classification: will the customer buy again?")
    cm = res["classification"]["table"].loc[res["classification"]["best"]]
    st.markdown(f'<div class="note">Predicts whether a customer will place <b>at least one order in the next {meta["horizon_days"]} days</b>. '
                f'Model: <b>{art["models"]["names"]["classification"]}</b> - test accuracy {cm["Accuracy"]:.1%}, ROC-AUC {cm["ROC-AUC"]:.3f}.</div>', unsafe_allow_html=True)
    vals, ok, row = customer_form("clf")
    if st.button("Predict", type="primary", disabled=not ok):
        try:
            out = B.predict_one(art, vals)
            a, b = st.columns([1, 2])
            a.metric("Prediction", "Will buy again" if out["prob_active"] >= .5 else "Unlikely to buy")
            if row is not None:
                a.metric("Actual outcome", "Bought" if row["Active"] else "Did not buy")
            fig = px.bar(x=["Won't buy", "Will buy"], y=out["proba"], color=["Won't buy", "Will buy"], color_discrete_sequence=["#E0457B", "#00A98F"],
                         labels={"x": "", "y": "Probability"}, title="Class probabilities", range_y=[0, 1], text=[f"{p:.1%}" for p in out["proba"]])
            b.plotly_chart(fig_style(fig.update_layout(showlegend=False), 300), width="stretch")
            st.info(f"Segment: **{out['segment_name']}** - {B.action_for(out['segment_name'])}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page == "Model Comparison":
    st.title("Model Comparison")
    tab1, tab2 = st.tabs(["Classification", "Regression"])
    with tab1:
        ct = res["classification"]["table"]
        st.caption(f"Test set: {meta['n_test']:,} customers. CV AUC = 5-fold stratified cross-validation on the training set. Best (by CV AUC): **{res['classification']['best']}**")
        st.dataframe(ct.style.format("{:.3f}").highlight_max(axis=0, color="#cfe8ff"), width="stretch")
        l, r = st.columns(2)
        fig = go.Figure()
        for n, (f, t_) in res["classification"]["roc"].items():
            fig.add_trace(go.Scatter(x=f, y=t_, mode="lines", name=f"{n} ({ct.loc[n, 'ROC-AUC']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False))
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        l.plotly_chart(fig_style(fig, 420), width="stretch")
        sel = r.selectbox("Confusion matrix for", list(ct.index), index=list(ct.index).index(res["classification"]["best"]))
        cmx = res["classification"]["confusion"][sel]
        r.plotly_chart(fig_style(px.imshow(cmx, text_auto=True, x=["Pred: no", "Pred: yes"], y=["Actual: no", "Actual: yes"],
                                           color_continuous_scale="Blues", title=f"Confusion matrix - {sel}"), 380), width="stretch")
        st.plotly_chart(fig_style(px.bar(res["classification"]["importance"].head(10), x="Importance", y="Feature", orientation="h",
                                         title=f"Permutation importance (AUC drop) - {res['classification']['best']}").update_yaxes(autorange="reversed")), width="stretch")
    with tab2:
        rt = res["regression"]["table"]
        st.caption("Metrics in GBP on the test set. Models are fitted to log(1 + spend); 'R2 (log scale)' measures fit on that scale. "
                   f"Best (by RMSE): **{res['regression']['best']}**")
        st.dataframe(rt.style.format("{:,.3f}"), width="stretch")
        st.caption("Linear models can have negative R2 in GBP: back-transforming log predictions amplifies a few large over-predictions, even though they beat the baseline on the log scale.")
        sel = st.selectbox("Actual vs predicted for", [m for m in rt.index], index=list(rt.index).index(res["regression"]["best"]))
        y = res["regression"]["test"]["Actual"].values
        p = res["regression"]["all_preds"][sel]
        l, r = st.columns(2)
        fig = px.scatter(x=y, y=p, opacity=.5, log_x=True, log_y=True, labels={"x": "Actual", "y": "Predicted"}, title=f"Actual vs predicted - {sel}")
        fig.add_trace(go.Scatter(x=[1, y.max()], y=[1, y.max()], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False))
        l.plotly_chart(fig_style(fig), width="stretch")
        r.plotly_chart(fig_style(px.histogram(x=np.log1p(p) - np.log1p(y), nbins=50, labels={"x": "log(1+pred) - log(1+actual)"}, title="Residuals (log scale)")), width="stretch")
        st.plotly_chart(fig_style(px.bar(rt["RMSE"].reset_index(), x="Model", y="RMSE", title="Test RMSE by model (GBP, lower is better)")), width="stretch")

elif page == "Model Insights":
    st.title("Model Insights & Recommendations")
    prof = B.segment_profile(cust, cust["Segment"].values)
    prof.insert(0, "Name", [seg["names"][i] for i in prof.index])
    ks = res["clustering"]["k_scores"]
    top = prof.sort_values("% Revenue", ascending=False).iloc[0]
    low = prof.sort_values("% Customers", ascending=False).iloc[0]
    st.markdown(f"""
**Headline findings (computed from this dataset)**
- The **{top['Name']}** segment is **{top['% Customers']:.0f}%** of customers but **{top['% Revenue']:.0f}%** of historic revenue, and **{top['Repeat rate next 90d (%)']:.0f}%** of them bought again in the next {meta['horizon_days']} days.
- The largest segment by headcount is **{low['Name']}** ({low['% Customers']:.0f}% of customers, {low['% Revenue']:.0f}% of revenue; {low['Repeat rate next 90d (%)']:.0f}% repeat rate).
- Overall, {100*meta['active_rate']:.0f}% of customers purchased again within {meta['horizon_days']} days.
- Top drivers of repeat purchase: **{', '.join(res['classification']['importance'].head(3)['Feature'])}**. Top drivers of future spend: **{', '.join(res['regression']['importance'].head(3)['Feature'])}**.
""")
    l, r = st.columns(2)
    l.plotly_chart(fig_style(px.bar(prof, x="Name", y=["% Customers", "% Revenue"], barmode="group", title="Customers vs revenue share by segment").update_layout(xaxis_title="", yaxis_title="%")), width="stretch")
    r.plotly_chart(fig_style(px.bar(prof, x="Name", y="Repeat rate next 90d (%)", color="Name", color_discrete_map=seg_colors,
                                    title=f"Share who purchased in the next {meta['horizon_days']} days").update_layout(showlegend=False, xaxis_title="")), width="stretch")
    st.subheader("Segment playbook")
    for i, row in prof.iterrows():
        with st.expander(f"{row['Name']}  -  {int(row['Customers']):,} customers, {row['% Revenue']:.0f}% of revenue", expanded=i == 0):
            st.write(f"Typical customer: last order **{row['Recency']:.0f} days** ago, **{row['Frequency']:.0f} orders**, **{gbp(row['Monetary'])}** lifetime spend, "
                     f"**{gbp(row['AvgOrderValue'])}** per order. Average spend in the next {meta['horizon_days']} days: **{gbp(row['Avg future spend (GBP)'])}**.")
            st.write("**Recommended action:** " + B.action_for(row["Name"]))
    st.subheader("Where the model can and cannot help")
    st.markdown(f"""
- **{res['classification']['importance'].iloc[0]['Feature']}** is the single most important feature of the repeat-purchase model (permutation importance {res['classification']['importance'].iloc[0]['Importance']:.3f} AUC drop vs {res['classification']['importance'].iloc[1]['Importance']:.3f} for the next feature), so a simple trigger on it is a natural basis for retention campaigns.
- **Spend is hard to predict precisely** (R² {res['regression']['table'].loc[res['regression']['best'], 'R2']:.2f}; RMSE {gbp(res['regression']['table'].loc[res['regression']['best'], 'RMSE'])}): a few wholesale customers dominate revenue. Use predictions to rank and prioritise accounts, not as exact forecasts.
- Silhouette scores for k = 2 to 8 are shown below. The chosen k = {seg['k']} balances cluster quality with business usefulness (k = 2 only separates active from dormant).
""")
    st.plotly_chart(fig_style(px.line(ks, x="k", y="Silhouette", markers=True, title="Silhouette score by number of clusters"), 300), width="stretch")
    st.caption("Limitations: single snapshot split (not rolling), the data covers only customers with a Customer ID (~78% of lines), and the model uses no marketing or product-category information.")
