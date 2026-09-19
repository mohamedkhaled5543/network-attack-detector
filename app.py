import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Network Attack Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    pre = joblib.load("preprocessor.pkl")
    model = joblib.load("model.pkl")
    skewed_cols = joblib.load("skewed_cols.pkl")
    return pre, model, skewed_cols


pre, model, skewed_cols = load_artifacts()
required = list(pre.feature_names_in_)


def prepare(df):
    """Same steps as training: strip names, select features, log1p skewed cols."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, df, missing
    X = df[required].copy()
    cols = [c for c in skewed_cols if c in X.columns]
    X[cols] = np.log1p(X[cols])
    return X, df, []


st.title("🛡️ Network Attack Detector")
st.caption("XGBoost model trained on UNSW-NB15. Upload flow records. Get attack predictions.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Attack threshold", 0.1, 0.9, 0.5, 0.05)
    st.write("Higher = fewer false alarms, more missed attacks.")
    with st.expander("Required columns"):
        st.code(", ".join(required))

file = st.file_uploader("Upload a CSV of network flows", type="csv")

if file is None:
    st.info("Upload a CSV to start. Use sample_flows.csv from the repo to try it.")
    st.stop()

raw = pd.read_csv(file)
X, df, missing = prepare(raw)

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

valid = X.notna().all(axis=1)
if (~valid).any():
    st.warning(f"Dropped {(~valid).sum()} rows with missing values.")
X = X[valid]
df = df.loc[X.index]

proba = model.predict_proba(pre.transform(X))[:, 1]
pred = (proba > threshold).astype(int)

out = df.copy()
out["attack_probability"] = proba.round(4)
out["prediction"] = np.where(pred == 1, "Attack", "Normal")

c1, c2, c3 = st.columns(3)
c1.metric("Flows analyzed", f"{len(out):,}")
c2.metric("Flagged as attack", f"{int(pred.sum()):,}")
c3.metric("Attack rate", f"{pred.mean():.1%}")

if "label" in out.columns:
    acc = (pred == out["label"].astype(int).values).mean()
    st.metric("Accuracy vs. true labels", f"{acc:.1%}")

left, right = st.columns([1, 2])
with left:
    st.subheader("Prediction split")
    st.bar_chart(out["prediction"].value_counts())
with right:
    st.subheader("Highest-risk flows")
    st.dataframe(
        out.sort_values("attack_probability", ascending=False).head(50),
        use_container_width=True,
    )

st.download_button(
    "Download all predictions",
    out.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)
