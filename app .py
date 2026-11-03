"""
Trading Risk Monitor - local dashboard
Run with:  streamlit run app.py
"""
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
FEATURES = ["Entry_Exit_Deviation", "Return_Percentage", "Position_Size_Deviation",
            "Holding_Period_Deviation", "Volume_Anomaly"]
FEATURE_LABELS = {
    "Entry_Exit_Deviation": "Price move %",
    "Return_Percentage": "Return %",
    "Position_Size_Deviation": "Position size (z)",
    "Holding_Period_Deviation": "Holding period (z)",
    "Volume_Anomaly": "Volume vs recent (z)",
}
REQUIRED = ["Entry Price", "Exit Price", "Returns (₹)", "Holding Period (Days)", "Quantity/Volume"]

st.set_page_config(page_title="Trading Risk Monitor", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .kpi {background: linear-gradient(135deg,#1f2a44,#2d3f66); border-radius: 14px;
          padding: 18px 20px; color: #fff; box-shadow: 0 4px 14px rgba(0,0,0,.15);}
    .kpi .label {font-size: 0.82rem; opacity: .75; letter-spacing: .03em; text-transform: uppercase;}
    .kpi .value {font-size: 1.9rem; font-weight: 700; margin-top: 4px;}
    .kpi .sub {font-size: 0.8rem; opacity: .7; margin-top: 2px;}
    .hero {background: linear-gradient(120deg,#0f1b33,#24406e); color:#fff; padding: 22px 26px;
           border-radius: 16px; margin-bottom: 18px;}
    .hero h1 {margin: 0; font-size: 1.7rem;}
    .hero p {margin: 4px 0 0 0; opacity: .8;}
    .warn {background:#fff7e6; border-left:4px solid #f0a500; padding: 10px 14px; border-radius: 8px;
           color:#5c4300; font-size: 0.9rem;}
    .tag-risky {background:#ffe3e3; color:#b42318; padding:2px 8px; border-radius:999px; font-weight:600;}
    .tag-normal {background:#e3f7ec; color:#067647; padding:2px 8px; border-radius:999px; font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data
def load_trades(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Trade Book")
    df = df.dropna(subset=REQUIRED).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Entry_Exit_Deviation"] = abs((df["Exit Price"] - df["Entry Price"]) / df["Entry Price"]) * 100
    df["Return_Percentage"] = df["Returns (₹)"] / (df["Entry Price"] * df["Quantity/Volume"]) * 100
    df["Position_Size_Deviation"] = (df["Quantity/Volume"] - df["Quantity/Volume"].mean()) / df["Quantity/Volume"].std()
    df["Holding_Period_Deviation"] = (df["Holding Period (Days)"] - df["Holding Period (Days)"].mean()) / df["Holding Period (Days)"].std()
    roll = df["Quantity/Volume"].rolling(50, min_periods=1)
    df["Volume_Anomaly"] = (df["Quantity/Volume"] - roll.mean()) / (roll.std() + 1)

    # Warning signs (0 or 1 each) and the composite score
    df["W_PoorReturn"] = (df["Return_Percentage"] <= df["Return_Percentage"].quantile(0.25)).astype(int)
    df["W_HighMove"] = (df["Entry_Exit_Deviation"] >= df["Entry_Exit_Deviation"].quantile(0.75)).astype(int)
    df["W_Volume"] = (df["Volume_Anomaly"].abs() > 1.5).astype(int)
    df["W_Holding"] = (df["Holding_Period_Deviation"].abs() > 1.5).astype(int)
    df["W_Size"] = (df["Position_Size_Deviation"].abs() > 1.5).astype(int)
    df["Risk_Score"] = df[["W_PoorReturn", "W_HighMove", "W_Volume", "W_Holding", "W_Size"]].sum(axis=1)
    df["Is_Risky"] = (df["Risk_Score"] >= 2).astype(int)
    return df


# ----------------------------------------------------------------- models
@st.cache_resource(show_spinner=False)
def train_and_select(file_bytes: bytes):
    """Same logic as the notebook: compare 5 models, keep top 2 by CV F1, pick final by test F1."""
    df = engineer_features(load_trades(file_bytes))
    X = StandardScaler().fit_transform(df[FEATURES].fillna(0))
    y = df["Is_Risky"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                                        random_state=RANDOM_STATE),
    }

    cv_rows, fitted, test_rows = [], {}, []
    for name, m in models.items():
        res = cross_validate(m, X_tr, y_tr, cv=cv, scoring=["f1", "precision", "recall", "roc_auc"])
        cv_rows.append({"Model": name, "CV F1": res["test_f1"].mean(), "CV F1 std": res["test_f1"].std(),
                        "CV Precision": res["test_precision"].mean(), "CV Recall": res["test_recall"].mean(),
                        "CV ROC-AUC": res["test_roc_auc"].mean()})
        m.fit(X_tr, y_tr)
        fitted[name] = m
        p = m.predict(X_te)
        pr = m.predict_proba(X_te)[:, 1]
        test_rows.append({"Model": name, "Test F1": f1_score(y_te, p), "Test Precision": precision_score(y_te, p),
                          "Test Recall": recall_score(y_te, p), "Test ROC-AUC": roc_auc_score(y_te, pr),
                          "cm": confusion_matrix(y_te, p).tolist()})

    cv_table = pd.DataFrame(cv_rows).sort_values("CV F1", ascending=False).reset_index(drop=True)
    test_table = pd.DataFrame(test_rows)
    top2 = cv_table["Model"].head(2).tolist()
    final_name = test_table[test_table["Model"].isin(top2)].sort_values("Test F1", ascending=False).iloc[0]["Model"]

    # Permutation importance for the final model on the test set
    imp = permutation_importance(fitted[final_name], X_te, y_te, scoring="f1",
                                 n_repeats=20, random_state=RANDOM_STATE).importances_mean
    importance = pd.Series(imp, index=FEATURES).sort_values(ascending=True)

    # Scores for every trade (scaler fit on all trades, same as the notebook)
    all_proba = fitted[final_name].predict_proba(X)[:, 1]

    # Isolation Forest anomaly flags
    iso = IsolationForest(n_estimators=100, contamination=0.10, random_state=RANDOM_STATE)
    anomaly = (iso.fit_predict(X) == -1).astype(int)

    return {
        "df": df, "cv_table": cv_table, "test_table": test_table.drop(columns=["cm"]),
        "cm_final": np.array(test_table.loc[test_table["Model"] == final_name, "cm"].iloc[0]),
        "top2": top2, "final_name": final_name, "importance": importance,
        "proba": all_proba, "anomaly": anomaly, "y_te": y_te,
        "final_test": test_table[test_table["Model"] == final_name].iloc[0],
    }


def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 📂 Your trade book")
    uploaded = st.file_uploader("Upload Tradebook-Kite.xlsx", type=["xlsx"])
    st.caption("Needs the 'Trade Book' sheet from Zerodha Kite.")
    st.markdown("---")
    st.markdown("### 🎛️ Filters")
    risk_filter = st.radio("Show trades", ["All", "Risky only", "Normal only"], horizontal=False)
    anomaly_only = st.checkbox("Isolation Forest anomalies only", value=False)
    threshold = st.slider("Alert threshold (risk probability)", 0.05, 0.95, 0.50, 0.05,
                          help="Trades with a risk probability at or above this value are flagged.")

if uploaded is None:
    st.markdown('<div class="hero"><h1>📊 Trading Risk Monitor</h1>'
                '<p>Upload your Kite trade book in the sidebar to start.</p></div>', unsafe_allow_html=True)
    st.info("Tip: you can use the same Tradebook-Kite.xlsx file from the project.")
    st.stop()

file_bytes = uploaded.getvalue()
with st.spinner("Training 5 models and scoring your trades..."):
    R = train_and_select(file_bytes)

df = R["df"].copy()
df["Risk_Probability"] = R["proba"]
df["Anomaly_IF"] = R["anomaly"]
df["Alert"] = (df["Risk_Probability"] >= threshold).astype(int)

st.markdown(f'<div class="hero"><h1>📊 Trading Risk Monitor</h1>'
            f'<p>Final model: <b>{R["final_name"]}</b> &nbsp;•&nbsp; {len(df):,} trades analysed '
            f'&nbsp;•&nbsp; {df["Symbol/Ticker"].nunique()} symbols</p></div>', unsafe_allow_html=True)

st.markdown('<div class="warn">⚠️ The risky label is built from the same five features the model uses. '
            'These scores show how well the model reproduces our risk rule. They are not proof of predicting future losses.</div>',
            unsafe_allow_html=True)
st.write("")

# ----------------------------------------------------------------- KPIs
c1, c2, c3, c4, c5 = st.columns(5)
with c1: kpi("Trades", f"{len(df):,}", f"{df['Symbol/Ticker'].nunique()} symbols")
with c2: kpi("Risky trades", f"{df['Is_Risky'].mean()*100:.1f}%", f"{df['Is_Risky'].sum()} trades")
with c3: kpi("Isolation Forest", f"{df['Anomaly_IF'].sum()}", f"{df.loc[df.Anomaly_IF==1,'Is_Risky'].mean()*100:.0f}% also risky")
with c4: kpi("Test F1", f"{R['final_test']['Test F1']:.3f}", "held-out 30%")
with c5: kpi("Test ROC-AUC", f"{R['final_test']['Test ROC-AUC']:.3f}", "held-out 30%")
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🧪 Model comparison", "🔎 Trade explorer", "ℹ️ How it works"])

# ----------------------------------------------------------------- Overview
with tab1:
    left, right = st.columns([1.15, 1])
    with left:
        daily = df.assign(Date=pd.to_datetime(df["Entry Date"]).dt.to_period("M").dt.to_timestamp()) \
                  .groupby("Date").agg(Trades=("Is_Risky", "size"), Risky=("Is_Risky", "sum")).reset_index()
        fig = go.Figure()
        fig.add_bar(x=daily["Date"], y=daily["Trades"], name="All trades", marker_color="#8fa8d6")
        fig.add_bar(x=daily["Date"], y=daily["Risky"], name="Risky", marker_color="#e5484d")
        fig.update_layout(barmode="overlay", title="Trades per month", height=360,
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.pie(values=[(df.Is_Risky == 0).sum(), (df.Is_Risky == 1).sum()],
                     names=["Normal", "Risky"], hole=0.6,
                     color_discrete_sequence=["#30a46c", "#e5484d"], title="Risky vs normal")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        sym = df.groupby("Symbol/Ticker").agg(Trades=("Is_Risky", "size"), RiskyPct=("Is_Risky", "mean"),
                                              PnL=("Returns (₹)", "sum")).reset_index().sort_values("Trades", ascending=False).head(12)
        fig = px.bar(sym, x="Symbol/Ticker", y="Trades", color="RiskyPct", color_continuous_scale="RdYlGn_r",
                     title="Top symbols (colour = share of risky trades)")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.histogram(df, x="Risk_Score", color="Is_Risky", nbins=6,
                           color_discrete_map={0: "#30a46c", 1: "#e5484d"}, title="Warning signs per trade")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), bargap=0.15)
        st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------- Model comparison
with tab2:
    st.subheader("Five algorithms, same data, same split")
    cvt = R["cv_table"].copy()
    tt = R["test_table"].copy()
    combined = cvt.merge(tt, on="Model")
    combined["Selected"] = combined["Model"].apply(lambda m: "⭐ Final" if m == R["final_name"]
                                                  else ("✓ Top 2" if m in R["top2"] else ""))
    st.dataframe(combined[["Model", "CV F1", "CV F1 std", "CV Recall", "CV ROC-AUC",
                           "Test F1", "Test Precision", "Test Recall", "Test ROC-AUC", "Selected"]]
                 .style.format({c: "{:.3f}" for c in combined.columns if c not in ("Model", "Selected")}),
                 width="stretch", hide_index=True)

    left, right = st.columns([1.2, 1])
    with left:
        fig = go.Figure()
        fig.add_bar(x=combined["Model"], y=combined["CV F1"], name="CV F1",
                    error_y=dict(type="data", array=combined["CV F1 std"]), marker_color="#8fa8d6")
        fig.add_bar(x=combined["Model"], y=combined["Test F1"], name="Test F1", marker_color="#24406e")
        fig.update_layout(barmode="group", title="F1 by model", height=380, yaxis_range=[0, 1.05],
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
    with right:
        cm = R["cm_final"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred normal", "Pred risky"], y=["Actual normal", "Actual risky"],
                        title=f"Confusion matrix: {R['final_name']}")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    st.success(f"Top two by cross-validated F1: **{R['top2'][0]}** and **{R['top2'][1]}**. "
               f"Final model by held-out F1: **{R['final_name']}**.")

# ----------------------------------------------------------------- Trade explorer
with tab3:
    view = df.copy()
    if risk_filter == "Risky only":
        view = view[view.Is_Risky == 1]
    elif risk_filter == "Normal only":
        view = view[view.Is_Risky == 0]
    if anomaly_only:
        view = view[view.Anomaly_IF == 1]

    st.caption(f"Showing {len(view):,} trades. Alerts at probability ≥ {threshold:.2f}: "
               f"**{int(view['Alert'].sum())}**")

    cols = ["Entry Date", "Symbol/Ticker", "Quantity/Volume", "Entry Price", "Exit Price",
            "Returns (₹)", "Holding Period (Days)", "Risk_Score", "Risk_Probability", "Alert", "Anomaly_IF"]
    st.dataframe(view[cols].sort_values("Risk_Probability", ascending=False).style.format(
        {"Risk_Probability": "{:.2%}", "Entry Price": "{:.2f}", "Exit Price": "{:.2f}", "Returns (₹)": "{:,.0f}"}),
        width="stretch", hide_index=True, height=360)

    st.markdown("#### Inspect one trade")
    idx = st.selectbox("Pick a trade (row number)", df.index,
                       format_func=lambda i: f"#{i} • {df.loc[i,'Symbol/Ticker']} • {str(df.loc[i,'Entry Date'])[:10]} • "
                                             f"risk {df.loc[i,'Risk_Probability']:.0%}")
    row = df.loc[idx]
    a, b = st.columns([1, 1.2])
    with a:
        status = "RISKY" if row.Is_Risky else "NORMAL"
        css = "tag-risky" if row.Is_Risky else "tag-normal"
        st.markdown(f"**Verdict:** <span class='{css}'>{status}</span>", unsafe_allow_html=True)
        st.metric("Model risk probability", f"{row.Risk_Probability:.1%}")
        st.metric("Warning signs", f"{int(row.Risk_Score)} of 5")
        st.metric("Return", f"₹{row['Returns (₹)']:,.0f}", f"{row.Return_Percentage:.2f}%")
        st.write("Isolation Forest:", "🚨 unusual trade" if row.Anomaly_IF else "✅ typical trade")
    with b:
        names = {"W_PoorReturn": "Poor return (bottom 25%)", "W_HighMove": "Big price move (top 25%)",
                 "W_Volume": "Unusual volume", "W_Holding": "Unusual holding period", "W_Size": "Unusual position size"}
        checks = pd.DataFrame({"Warning sign": list(names.values()),
                               "Triggered": [int(row[k]) for k in names]})
        fig = px.bar(checks, x="Triggered", y="Warning sign", orientation="h", range_x=[0, 1],
                     color="Triggered", color_continuous_scale=["#e3f7ec", "#e5484d"])
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False,
                          title="Which warning signs fired")
        st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------- How it works
with tab4:
    st.markdown("""
**Pipeline**
1. **Clean** the trade book and drop incomplete rows.
2. **Engineer five features** per trade: price move, return %, position size (z), holding period (z), and volume vs the last 50 trades (z).
3. **Score warning signs.** One point each for poor return, big price move, unusual volume, unusual holding period and unusual size. Two or more points = risky.
4. **Compare five algorithms** with 5-fold cross-validation on the training 70%. Keep the top two by F1, then choose the final model on the held-out 30%.
5. **Flag anomalies** with Isolation Forest (the most unusual 10% of trades).
6. **Alert** on any trade whose risk probability passes the threshold you set in the sidebar.

**Limits**
- The risk label comes from the same features the models learn from, so scores show agreement with our rule, not real future losses.
- The 10% anomaly rate is a setting we chose, not something the data proved.
- This is a study prototype. It is not financial advice and is not connected to live trading.
""")

