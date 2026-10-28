import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"

st.set_page_config(page_title="Farmer Resilience Predictor", layout="wide")


@st.cache_resource
def load():
    model = joblib.load(MODEL_DIR / "ebm.joblib")
    meta = json.load(open(MODEL_DIR / "meta.json"))
    return model, meta


ebm, meta = load()
cols, num, cat = meta["columns"], meta["numeric"], meta["categorical"]

LABELS = {
    "Total Land extent": {1: "Less than 1 acre", 2: "1-2 acres", 3: "Above 2 acres"},
    "Farming experience": {1: "Less than 5 years", 2: "5-10 years", 3: "More than 10 years"},
    "Monthly income from farming": {1: "Less than 50,000 LKR", 2: "50,000-100,000 LKR", 3: "More than 100,000 LKR"},
    "Household size": {1: "3 or fewer", 2: "4-6", 3: "More than 6"},
    "Education level of the farmer": {1: "None", 2: "Primary", 3: "Secondary", 4: "Higher"},
}
CH_LABELS = {1: "Not a challenge", 2: "Light", 3: "Moderate", 4: "Considerable", 5: "Significant"}
RM_LABELS = {0: "Not practiced", 1: "Rarely", 2: "Moderately", 3: "Well practiced"}

values = {}


def widget(c, key):
    """Render one input for feature c and store its value."""
    if c in cat:
        values[c] = st.selectbox(c, cat[c], index=len(cat[c]) - 1 if c == "Ownership of the land" else 0, key=key)
        return
    lo, hi, med = num[c]
    lo, hi, med = int(lo), int(hi), int(round(med))
    if c.startswith("CH: "):
        values[c] = st.select_slider(c[4:], options=list(CH_LABELS), value=med, format_func=CH_LABELS.get, key=key)
    elif c.startswith("RM: "):
        values[c] = st.select_slider(c[4:], options=list(RM_LABELS), value=med, format_func=RM_LABELS.get, key=key)
    elif c in LABELS:
        opts = list(LABELS[c])
        values[c] = st.selectbox(c, opts, index=opts.index(med), format_func=LABELS[c].get, key=key)
    elif hi <= 1:
        values[c] = int(st.selectbox(c, ["No", "Yes"], index=med, key=key) == "Yes")
    else:
        values[c] = st.slider(c, lo, hi, med, key=key)


st.title("Smallholder Farmer Resilience Capacity Predictor")
st.caption("Explainable Boosting Machine trained on 300 Sri Lankan smallholder farmers. "
           "Predicts High vs Low self-assessed resilience capacity. Research prototype, not for real decisions.")

profile = [c for c in cols if not c.startswith(("CH: ", "RM: "))]
challenges = [c for c in cols if c.startswith("CH: ")]
strategies = [c for c in cols if c.startswith("RM: ")]

t1, t2, t3 = st.tabs(["Farmer profile", "Challenges faced", "Risk-management strategies"])
with t1:
    grid = st.columns(3)
    for i, c in enumerate(profile):
        with grid[i % 3]:
            widget(c, f"p{i}")
with t2:
    grid = st.columns(2)
    for i, c in enumerate(challenges):
        with grid[i % 2]:
            widget(c, f"c{i}")
with t3:
    grid = st.columns(2)
    for i, c in enumerate(strategies):
        with grid[i % 2]:
            widget(c, f"r{i}")

if st.button("Predict", type="primary"):
    row = pd.DataFrame([values])[cols]
    p_high = float(ebm.predict_proba(row)[0, 1])
    label = "High" if p_high >= 0.5 else "Low"

    a, b = st.columns([1, 2])
    a.metric("Predicted resilience capacity", label)
    a.metric("Probability of High", f"{p_high:.1%}")
    a.progress(p_high)

    d = ebm.explain_local(row).data(0)
    s = pd.Series(d["scores"], index=d["names"])
    s = s.reindex(s.abs().sort_values().index).tail(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s.index, s.values, color=["tab:red" if v < 0 else "tab:blue" for v in s.values])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Contribution to log-odds of High (blue = pushes up, red = pushes down)")
    ax.set_title("Top 10 factors for this farmer")
    plt.tight_layout()
    b.pyplot(fig)

with st.expander("Global feature importance (whole model)"):
    imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values().tail(15)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.barh(imp.index, imp.values)
    ax2.set_xlabel("Mean |score|")
    plt.tight_layout()
    st.pyplot(fig2)
