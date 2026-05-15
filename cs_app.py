"""Streamlit app — compressive strength prediction for geopolymer concrete.

Loads the app champion (ExtraTrees trained with GroupKFold-honest CV) from
cs_champions/cs_app_champion.pkl. User enters raw mix-design inputs;
app derives the 9 engineered features the model was trained on, then predicts
28-day compressive strength in MPa.

Run:
    streamlit run cs_app.py
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
CHAMPION_PATH = os.path.join(ROOT, "cs_champions", "cs_app_champion.pkl")
MANIFEST_PATH = os.path.join(ROOT, "cs_champions", "manifest.json")


@st.cache_resource
def load_champion():
    if not os.path.exists(CHAMPION_PATH):
        st.error(
            f"App champion not found at {CHAMPION_PATH}. "
            "Run the CS notebook's Phase 8 cell to generate it."
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
# the CS app champion was trained on (engineered_9 = engineered_12 minus the
# bottom-3 permutation-importance features: activator_binder_ratio, rca_fraction,
# r_sand_fraction).
# -----------------------------------------------------------------------------
def derive_features(raw: dict) -> pd.DataFrame:
    eps = 1e-9
    total_binder = raw["flyash"] + raw["ggbfs"]
    total_activator = raw["na2sio3"] + raw["naoh"]
    feats = {
        "total_binder": total_binder,
        "ggbfs_fraction": raw["ggbfs"] / (total_binder + eps),
        "total_activator": total_activator,
        "na2sio3_naoh_ratio": raw["na2sio3"] / (raw["naoh"] + eps),
        "naoh_molarity": raw["naoh_molarity"],
        "w_b_ratio": raw["water"] / (total_binder + eps),
        "curing_temp_c": raw["curing_temp_c"],
        "curing_time_hr": raw["curing_time_hr"],
        "sp": raw["sp"],
    }
    return pd.DataFrame([feats])


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Compressive Strength Predictor — Geopolymer Concrete",
                   page_icon="🏗️", layout="wide",
                   initial_sidebar_state="collapsed")

art, manifest = load_champion()
model = art["estimator"]
model_name = art.get("model_name", "Unknown")
feature_order = art["features"]
app_meta = manifest.get("app", {})

st.title("Compressive Strength Predictor — Geopolymer Concrete")
st.caption("Trained on 353 mix designs with GroupKFold-honest cross-validation. "
           "Predicts 28-day compressive strength.")

st.markdown("### Mix design inputs")
st.caption("Enter raw quantities per cubic metre of concrete. "
           "The app derives the 9 engineered features the model expects.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Binders (kg/m³)**")
    flyash = st.number_input("Fly ash", min_value=0.0, max_value=700.0, value=400.0, step=10.0)
    ggbfs = st.number_input("GGBFS", min_value=0.0, max_value=500.0, value=0.0, step=10.0)
    st.markdown("**Activator (kg/m³)**")
    na2sio3 = st.number_input("Na₂SiO₃ (sodium silicate)", min_value=0.0, max_value=300.0, value=117.0, step=5.0)
    naoh = st.number_input("NaOH (sodium hydroxide)", min_value=0.0, max_value=240.0, value=53.0, step=5.0)
    naoh_molarity = st.slider("NaOH molarity (M)", min_value=0.0, max_value=20.0, value=12.0, step=0.5)

with col2:
    st.markdown("**Water + Superplasticizer**")
    water = st.number_input("Water (kg/m³)", min_value=0.0, max_value=130.0, value=0.0, step=5.0)
    sp = st.slider("Superplasticizer (kg/m³)", min_value=0.0, max_value=25.0, value=0.0, step=0.5)
    st.markdown("**Aggregates (kg/m³)** (display only — model uses curing + binder/activator features)")
    nca = st.number_input("NCA (natural coarse aggregate)", min_value=0.0, max_value=1600.0, value=1190.0, step=50.0)
    rca = st.number_input("RCA (recycled coarse aggregate)", min_value=0.0, max_value=1250.0, value=0.0, step=50.0)
    sand = st.number_input("Sand", min_value=0.0, max_value=900.0, value=600.0, step=25.0)
    r_sand = st.number_input("R-sand (recycled sand)", min_value=0.0, max_value=550.0, value=0.0, step=25.0)

with col3:
    st.markdown("**Curing**")
    curing_temp_c = st.slider("Curing temperature (°C)", min_value=0, max_value=120, value=60, step=5)
    curing_time_hr = st.slider("Curing time (hr)", min_value=0, max_value=48, value=24, step=1)
    st.divider()
    st.markdown("**Derived features (for sanity)**")
    total_binder_preview = flyash + ggbfs
    total_activator_preview = na2sio3 + naoh
    total_agg_preview = nca + rca + sand + r_sand
    eps = 1e-9
    st.write(f"Total binder: **{total_binder_preview:.1f} kg/m³**")
    st.write(f"GGBFS fraction: **{ggbfs/(total_binder_preview+eps):.3f}**")
    st.write(f"Total activator: **{total_activator_preview:.1f} kg/m³**")
    st.write(f"Na₂SiO₃/NaOH ratio: **{na2sio3/(naoh+eps):.3f}**")
    st.write(f"w/b ratio: **{water/(total_binder_preview+eps):.3f}**")

st.divider()

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
predict_clicked = st.button("Predict compressive strength", type="primary", use_container_width=True)

if predict_clicked:
    raw_inputs = {
        "flyash": flyash, "ggbfs": ggbfs,
        "na2sio3": na2sio3, "naoh": naoh,
        "naoh_molarity": naoh_molarity,
        "water": water, "sp": sp,
        "nca": nca, "rca": rca, "sand": sand, "r_sand": r_sand,
        "curing_temp_c": curing_temp_c,
        "curing_time_hr": curing_time_hr,
    }
    X_features = derive_features(raw_inputs)
    X_features = X_features[feature_order]

    if flyash + ggbfs <= 0:
        st.error("Total binder (fly ash + GGBFS) must be greater than zero.")
    else:
        pred_mpa = float(model.predict(X_features.values)[0])
        pred_mpa_clamped = max(0.0, pred_mpa)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Predicted 28-day strength", f"{pred_mpa_clamped:.1f} MPa")
        with c2:
            # Compressive strength classification (common civil-engineering grades)
            if pred_mpa_clamped < 20:
                grade = "Below structural grade — non-load-bearing applications only"
            elif pred_mpa_clamped < 30:
                grade = "Normal-strength concrete (M20–M25 equivalent) — light structures, pavements"
            elif pred_mpa_clamped < 45:
                grade = "Medium-strength concrete (M30–M40) — typical RCC structures, beams, columns"
            elif pred_mpa_clamped < 60:
                grade = "High-strength concrete (M45–M55) — high-rise buildings, bridges"
            else:
                grade = "Very high-strength / ultra-high-performance — specialty structural elements"
            st.info(f"**Strength class**: {grade}")

        with st.expander("Engineered feature vector passed to the model"):
            st.dataframe(X_features.T.rename(columns={0: "value"}), use_container_width=True)

        # Reasonable-range check — warn if any raw input is far outside training distribution
        ranges = {
            "flyash": (0, 661), "ggbfs": (0, 500),
            "na2sio3": (0, 278), "naoh": (0, 232),
            "naoh_molarity": (0, 20),
            "water": (0, 122), "sp": (0, 24),
            "nca": (0, 1591), "rca": (0, 1225),
            "sand": (0, 875), "r_sand": (0, 525),
            "curing_temp_c": (0, 120),
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
    "⚠️ This model was trained on 353 mix designs. Predictions are most reliable for "
    "compositions similar to the training data (Fly-ash/GGBFS geopolymer concrete, "
    "alkali-activated, with or without recycled aggregates). Always validate critical mixes with bench testing."
)
