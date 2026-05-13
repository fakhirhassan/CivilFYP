"""Streamlit app — slump prediction for geopolymer concrete.

Loads the app champion (LightGBM trained with GroupKFold-honest CV) from
slump_champions/slump_app_champion.pkl. User enters raw mix-design inputs;
app derives the 9 engineered features the model was trained on, then predicts
slump in mm.

Run:
    streamlit run slump_app.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Paths + model loading
# -----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
CHAMPION_PATH = os.path.join(ROOT, "slump_champions", "slump_app_champion.pkl")
MANIFEST_PATH = os.path.join(ROOT, "slump_champions", "manifest.json")


@st.cache_resource
def load_champion():
    if not os.path.exists(CHAMPION_PATH):
        st.error(
            f"App champion not found at {CHAMPION_PATH}. "
            "Run the slump notebook's Phase 8 cell to generate it."
        )
        st.stop()
    art = joblib.load(CHAMPION_PATH)
    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        import json
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    return art, manifest


# -----------------------------------------------------------------------------
# Feature engineering — mirrors civil_utils.engineer_features for the 9 columns
# the app champion was trained on (feature_set='engineered_9').
# -----------------------------------------------------------------------------
def derive_features(raw: dict) -> pd.DataFrame:
    eps = 1e-9
    total_binder = raw["flyash"] + raw["ggbfs"]
    total_activator = raw["na2sio3"] + raw["naoh"]
    total_agg = raw["nca"] + raw["rca"] + raw["sand"] + raw["r_sand"]
    feats = {
        "total_binder": total_binder,
        "ggbfs_fraction": raw["ggbfs"] / (total_binder + eps),
        "total_activator": total_activator,
        "naoh_molarity": raw["naoh_molarity"],
        "activator_binder_ratio": raw["activator_binder_ratio"],
        "w_b_ratio": raw["water"] / (total_binder + eps),
        "rca_fraction": raw["rca"] / (total_agg + eps),
        "curing_time_hr": raw["curing_time_hr"],
        "sp": raw["sp"],
    }
    return pd.DataFrame([feats])


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Slump Predictor — Geopolymer Concrete",
                   page_icon="🧱", layout="wide",
                   initial_sidebar_state="collapsed")

art, manifest = load_champion()
model = art["estimator"]
model_name = art.get("model_name", "Unknown")
feature_order = art["features"]
app_meta = manifest.get("app", {})

st.title("Slump Predictor — Geopolymer Concrete")
st.caption("Trained on 84 mix designs with GroupKFold-honest cross-validation.")

st.markdown("### Mix design inputs")
st.caption("Enter raw quantities per cubic metre of concrete. "
           "The app derives the 9 engineered features the model expects.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Binders (kg/m³)**")
    flyash = st.number_input("Fly ash", min_value=0.0, max_value=600.0, value=300.0, step=10.0)
    ggbfs = st.number_input("GGBFS", min_value=0.0, max_value=600.0, value=100.0, step=10.0)
    st.markdown("**Activator (kg/m³)**")
    na2sio3 = st.number_input("Na₂SiO₃ (sodium silicate)", min_value=0.0, max_value=250.0, value=150.0, step=5.0)
    naoh = st.number_input("NaOH (sodium hydroxide)", min_value=0.0, max_value=150.0, value=65.0, step=5.0)
    naoh_molarity = st.slider("NaOH molarity (M)", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
    activator_binder_ratio = st.slider("Activator/binder ratio", min_value=0.0, max_value=3.0, value=0.55, step=0.05)

with col2:
    st.markdown("**Water + Superplasticizer**")
    water = st.number_input("Water (kg/m³)", min_value=0.0, max_value=200.0, value=30.0, step=5.0)
    sp = st.slider("Superplasticizer (kg/m³)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    st.markdown("**Aggregates (kg/m³)**")
    nca = st.number_input("NCA (natural coarse aggregate)", min_value=0.0, max_value=1500.0, value=500.0, step=50.0)
    rca = st.number_input("RCA (recycled coarse aggregate)", min_value=0.0, max_value=1500.0, value=400.0, step=50.0)
    sand = st.number_input("Sand", min_value=0.0, max_value=900.0, value=475.0, step=25.0)
    r_sand = st.number_input("R-sand (recycled sand)", min_value=0.0, max_value=800.0, value=0.0, step=25.0)

with col3:
    st.markdown("**Curing**")
    curing_time_hr = st.slider("Curing time (hr)", min_value=0, max_value=72, value=24, step=1)
    curing_temp_c = st.slider("Curing temperature (°C)", min_value=0, max_value=100, value=60, step=5,
                              help="Not used directly by the model — kept here for documentation only.")
    st.divider()
    st.markdown("**Derived features (for sanity)**")
    total_binder_preview = flyash + ggbfs
    total_activator_preview = na2sio3 + naoh
    total_agg_preview = nca + rca + sand + r_sand
    eps = 1e-9
    st.write(f"Total binder: **{total_binder_preview:.1f} kg/m³**")
    st.write(f"GGBFS fraction: **{ggbfs/(total_binder_preview+eps):.3f}**")
    st.write(f"Total activator: **{total_activator_preview:.1f} kg/m³**")
    st.write(f"w/b ratio: **{water/(total_binder_preview+eps):.3f}**")
    st.write(f"RCA fraction (of agg): **{rca/(total_agg_preview+eps):.3f}**")

st.divider()

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
predict_clicked = st.button("Predict slump", type="primary", use_container_width=True)

if predict_clicked:
    raw_inputs = {
        "flyash": flyash, "ggbfs": ggbfs,
        "na2sio3": na2sio3, "naoh": naoh,
        "naoh_molarity": naoh_molarity,
        "activator_binder_ratio": activator_binder_ratio,
        "water": water, "sp": sp,
        "nca": nca, "rca": rca, "sand": sand, "r_sand": r_sand,
        "curing_time_hr": curing_time_hr,
    }
    X_features = derive_features(raw_inputs)
    # Ensure column order matches training
    X_features = X_features[feature_order]

    if flyash + ggbfs <= 0:
        st.error("Total binder (fly ash + GGBFS) must be greater than zero.")
    else:
        pred_mm = float(model.predict(X_features.values)[0])
        pred_mm_int = int(round(max(0.0, pred_mm)))  # clamp negatives to 0, then round

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Predicted slump", f"{pred_mm_int} mm")
        with c2:
            # Slump classification (ACI / common civil convention)
            if pred_mm_int < 25:
                workability = "Very low — stiff mix, may need vibration/compaction effort"
            elif pred_mm_int < 50:
                workability = "Low — suitable for road slabs and foundations"
            elif pred_mm_int < 100:
                workability = "Medium — typical reinforced-concrete workability"
            elif pred_mm_int < 175:
                workability = "High — easy placing, pumpable mixes"
            else:
                workability = "Very high / flowing — self-consolidating mix territory"
            st.info(f"**Workability class**: {workability}")

        with st.expander("Engineered feature vector passed to the model"):
            st.dataframe(X_features.T.rename(columns={0: "value"}), use_container_width=True)

        # Reasonable-range check — warn if any raw input is far outside training distribution
        ranges = {
            "flyash": (0, 500), "ggbfs": (0, 500),
            "na2sio3": (0, 195), "naoh": (0, 108),
            "naoh_molarity": (0, 16), "activator_binder_ratio": (0, 2.73),
            "water": (0, 144), "sp": (0, 7.5),
            "nca": (0, 1261), "rca": (0, 1260),
            "sand": (0, 799), "r_sand": (0, 737),
            "curing_time_hr": (0, 48),
        }
        warnings = []
        for name, (lo, hi) in ranges.items():
            v = raw_inputs.get(name)
            if v is None: continue
            if v < lo or v > hi:
                warnings.append(f"**{name}** = {v} is outside training range [{lo}, {hi}]")
        if warnings:
            st.warning("⚠️ Inputs outside the training distribution — prediction may be unreliable:\n\n"
                       + "\n".join(f"- {w}" for w in warnings))

st.divider()
st.caption(
    "⚠️ This model was trained on 84 mix designs. Predictions are most reliable for "
    "compositions similar to the training data (Fly-ash/GGBFS geopolymer concrete, "
    "alkali-activated). Always validate critical mixes with bench testing."
)
