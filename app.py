"""
Streamlit app for Industry 4.0 Predictive Maintenance
1D-CNN + Siamese Network  |  18 hydraulic sensor channels
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make project src importable when running from the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Local model classes (no heavy project imports – only the model files)
# ---------------------------------------------------------------------------
from src.cnn_model import CNNModel
from src.siamese_model import SiameseNetwork

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 8   # timesteps per inference window
NUM_FEATURES = 18  # sensor channels

# Ordered list of the 18 feature columns (same order used in /predict)
FEATURE_KEYS = [
    "I_w_BLO_Weg",  "O_w_BLO_power",  "O_w_BLO_voltage",
    "I_w_BHL_Weg",  "O_w_BHL_power",  "O_w_BHL_voltage",
    "I_w_BHR_Weg",  "O_w_BHR_power",  "O_w_BHR_voltage",
    "I_w_BRU_Weg",  "O_w_BRU_power",  "O_w_BRU_voltage",
    "I_w_HR_Weg",   "O_w_HR_power",   "O_w_HR_voltage",
    "I_w_HL_Weg",   "O_w_HL_power",   "O_w_HL_voltage",
]

# Grouping of features by hydraulic module (for UI layout)
MODULES = {
    "BLO": ["I_w_BLO_Weg",  "O_w_BLO_power",  "O_w_BLO_voltage"],
    "BHL": ["I_w_BHL_Weg",  "O_w_BHL_power",  "O_w_BHL_voltage"],
    "BHR": ["I_w_BHR_Weg",  "O_w_BHR_power",  "O_w_BHR_voltage"],
    "BRU": ["I_w_BRU_Weg",  "O_w_BRU_power",  "O_w_BRU_voltage"],
    "HR":  ["I_w_HR_Weg",   "O_w_HR_power",   "O_w_HR_voltage"],
    "HL":  ["I_w_HL_Weg",   "O_w_HL_power",   "O_w_HL_voltage"],
}

# Sample values from the API schema example (the rest default to 0)
SAMPLE_VALUES = {
    "I_w_BLO_Weg":   -107,
    "I_w_BHR_Weg":  -1268,
    "I_w_BRU_Weg":    -26,
    "O_w_BRU_power":   84,
    "O_w_BRU_voltage": 11,
    "O_w_HR_power":  7168,
    "O_w_HR_voltage":  26,
    "O_w_HL_power":  7720,
    "O_w_HL_voltage":  24,
}

# Model file paths
CNN_MODEL_PATH     = PROJECT_ROOT / "models" / "cnn_model.pth"
SIAMESE_MODEL_PATH = PROJECT_ROOT / "models" / "siamese_model.pth"
RESULTS_DIR        = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Model loading  (cached so weights are only read once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CNN model …")
def load_cnn_model():
    """Load the trained CNN model onto CPU in eval mode."""
    checkpoint = torch.load(CNN_MODEL_PATH, map_location="cpu")
    model = CNNModel(
        input_shape=checkpoint["input_shape"],
        num_classes=checkpoint["num_classes"],
        config=checkpoint["config"],
        debug_mode=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading Siamese model …")
def load_siamese_model():
    """Load the trained Siamese network onto CPU in eval mode."""
    checkpoint = torch.load(SIAMESE_MODEL_PATH, map_location="cpu")
    model = SiameseNetwork(input_shape=(WINDOW_SIZE, NUM_FEATURES), debug_mode=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers  (mirror the /predict preprocessing exactly)
# ---------------------------------------------------------------------------

def build_tensor(features: list[float]) -> torch.Tensor:
    """
    Tile a single 18-value reading into a (1, 8, 18) window tensor.
    This replicates the preprocessing used in the FastAPI /predict endpoint.
    """
    arr = np.array(features, dtype=np.float32).reshape(1, 1, NUM_FEATURES)
    window = np.tile(arr, (1, WINDOW_SIZE, 1))   # shape → (1, 8, 18)
    return torch.FloatTensor(window)


def run_inference(features: list[float]):
    """
    Run both models on a feature vector and return:
      cnn_pred (0/1), cnn_confidence (float), siamese_similarity (float)

    Siamese forward(x1, x2) computes a sigmoid similarity in [0, 1].
    We pass the same window as both inputs (self-similarity), so a value
    close to 1 means the pattern is internally consistent (normal-ish),
    while a lower value can indicate anomaly.
    """
    cnn_model     = load_cnn_model()
    siamese_model = load_siamese_model()

    tensor = build_tensor(features)

    with torch.no_grad():
        # ── CNN ──────────────────────────────────────────────────────────
        cnn_out   = cnn_model(tensor)                        # (1, 2)
        cnn_probs = F.softmax(cnn_out, dim=-1)
        cnn_pred  = int(torch.argmax(cnn_probs, dim=-1).item())
        cnn_conf  = float(torch.max(cnn_probs).item())

        # ── Siamese: pass tensor as BOTH inputs to get a scalar similarity ─
        try:
            siam_out = siamese_model(tensor, tensor)         # (1, 1) sigmoid
            siam_sim = float(siam_out.squeeze().item())
        except Exception:
            siam_sim = cnn_conf   # fallback to CNN confidence

    return cnn_pred, cnn_conf, siam_sim


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------

def render_metric_card(label: str, value: str):
    """
    Render a robust HTML metric card that never fails due to adblockers
    or lazy-loaded dynamic JS chunk network issues (e.g. Metric.DMz1h4un.js).
    """
    st.markdown(
        f"""
        <div style="background-color: #1E2130; padding: 14px 18px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 8px;">
            <div style="color: #94A3B8; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
            <div style="color: #F8FAFC; font-size: 26px; font-weight: 700; margin-top: 4px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_prediction_result(cnn_pred: int, cnn_conf: float, siam_sim: float):
    """Render result banner and metric cards."""
    st.divider()
    if cnn_pred == 0:
        st.success("### ✅  Normal Operation")
    else:
        st.error("### ⚠️  Fault Detected")

    col1, col2 = st.columns(2)
    with col1:
        render_metric_card("CNN Confidence", f"{cnn_conf * 100:.1f} %")
    with col2:
        render_metric_card("Siamese Similarity", f"{siam_sim:.4f}")


# ── Page 1: Predict ─────────────────────────────────────────────────────────

def page_predict():
    st.header("🔮 Predict")

    tab_manual, tab_csv = st.tabs(["Manual Input", "CSV Upload"])

    # ── Manual tab ──────────────────────────────────────────────────────────
    with tab_manual:
        st.write("Enter sensor readings for a single time-step, then click **Predict**.")

        # "Load sample" pre-fills session state with the example values
        if st.button("📋 Load sample values"):
            for key in FEATURE_KEYS:
                st.session_state[f"inp_{key}"] = float(SAMPLE_VALUES.get(key, 0.0))

        # Render inputs grouped by module (two modules per row)
        module_names = list(MODULES.keys())
        values: dict[str, float] = {}

        for i in range(0, len(module_names), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j >= len(module_names):
                    break
                mod = module_names[i + j]
                with col:
                    st.subheader(f"Module {mod}")
                    for feat in MODULES[mod]:
                        # Use session_state value if set (e.g. by Load sample)
                        default = float(st.session_state.get(f"inp_{feat}", 0.0))
                        values[feat] = st.number_input(
                            feat,
                            value=default,
                            key=f"inp_{feat}",
                            step=1.0,
                            format="%.2f",
                        )

        if st.button("⚡ Predict", type="primary"):
            features = [values[k] for k in FEATURE_KEYS]
            with st.spinner("Running inference …"):
                cnn_pred, cnn_conf, siam_sim = run_inference(features)
            show_prediction_result(cnn_pred, cnn_conf, siam_sim)

    # ── CSV tab ─────────────────────────────────────────────────────────────
    with tab_csv:
        st.write(
            "Upload a CSV file with the 18 sensor columns. "
            "Predictions are made using a **sliding window of 8 rows**."
        )
        uploaded = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                return

            # Validate that all required columns are present
            missing = [c for c in FEATURE_KEYS if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                return

            df_features = df[FEATURE_KEYS].copy()
            n_rows = len(df_features)

            if n_rows < WINDOW_SIZE:
                st.warning(
                    f"Need at least {WINDOW_SIZE} rows for a sliding window "
                    f"(got {n_rows})."
                )
                return

            cnn_model     = load_cnn_model()
            siamese_model = load_siamese_model()

            results = []
            with st.spinner(f"Running sliding-window inference on {n_rows} rows …"):
                for start in range(n_rows - WINDOW_SIZE + 1):
                    window = df_features.iloc[start : start + WINDOW_SIZE].values  # (8, 18)
                    tensor = torch.FloatTensor(window).unsqueeze(0)  # (1, 8, 18)

                    with torch.no_grad():
                        cnn_out  = cnn_model(tensor)
                        probs    = F.softmax(cnn_out, dim=-1)
                        pred     = int(torch.argmax(probs, dim=-1).item())
                        conf     = float(torch.max(probs).item())

                        try:
                            # Pass tensor as both inputs → scalar sigmoid similarity
                            siam_out = siamese_model(tensor, tensor)
                            siam_sim = float(siam_out.squeeze().item())
                        except Exception:
                            siam_sim = conf

                    results.append({
                        "Window start": start,
                        "Window end":   start + WINDOW_SIZE - 1,
                        "Prediction":   "Fault" if pred == 1 else "Normal",
                        "CNN Confidence": f"{conf * 100:.1f} %",
                        "Siamese Similarity": f"{siam_sim:.4f}",
                    })

            results_df = pd.DataFrame(results)
            fault_pct  = (results_df["Prediction"] == "Fault").mean() * 100

            st.subheader("Results")
            st.dataframe(results_df, use_container_width=True)
            render_metric_card("Fault rate across windows", f"{fault_pct:.1f} %")


# ── Page 2: Model Results ────────────────────────────────────────────────────

def page_results():
    st.header("📊 Model Results")

    # ── Metrics table ────────────────────────────────────────────────────────
    metrics = pd.DataFrame({
        "Model":    ["CNN", "Siamese"],
        "Accuracy": [0.6411, 0.4260],
        "F1 Score": [0.5522, 0.3845],
    })
    st.subheader("Performance Metrics")
    st.dataframe(metrics.set_index("Model"), use_container_width=True)

    st.divider()

    # ── Result images ────────────────────────────────────────────────────────
    # Sorted list of all PNGs in results/ (top-level only; skip sub-dirs)
    image_files = sorted(RESULTS_DIR.glob("*.png"))

    if not image_files:
        st.info("No result images found in `results/`.")
        return

    st.subheader("Training & Evaluation Plots")

    # Display images two per row
    for i in range(0, len(image_files), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(image_files):
                break
            img_path = image_files[i + j]
            try:
                img = Image.open(img_path)
                col.image(img, caption=img_path.name, use_container_width=True)
            except Exception:
                col.warning(f"Could not load {img_path.name}")


# ── Page 3: About ────────────────────────────────────────────────────────────

def page_about():
    st.header("ℹ️ About")
    st.markdown(
        """
        ## Industry 4.0 Predictive Maintenance

        This application demonstrates **real-time fault detection** for hydraulic
        manufacturing equipment using two complementary deep-learning models:

        | Model | Architecture | Role |
        |-------|-------------|------|
        | **1D-CNN** | 3 × Conv1D + 3 × Dense | Primary classifier (Normal / Fault) |
        | **Siamese Network** | Shared 1D-CNN encoder + distance layer | Anomaly similarity scoring |

        ### Data
        - **18 sensor channels** across 6 hydraulic modules (BLO, BHL, BHR, BRU, HR, HL)
        - Each module exposes *Weg* (position), *power*, and *voltage* signals
        - Readings are grouped into **windows of 8 timesteps** before inference

        ### How to use
        1. Go to **Predict → Manual Input** to enter individual sensor readings
           and get an instant Normal / Fault decision.
        2. Go to **Predict → CSV Upload** to run batch inference with a sliding
           window over a longer sensor log.
        3. Go to **Model Results** to inspect training curves, confusion matrices,
           and other evaluation plots.

        ### Performance
        | Model   | Accuracy | F1 Score |
        |---------|----------|----------|
        | CNN     | 64.11 %  | 0.5522   |
        | Siamese | 42.60 %  | 0.3845   |

        > **Note:** The Siamese model is used as a *similarity / anomaly* indicator
        > rather than a standalone classifier.  The CNN prediction is the primary
        > output shown in the result banner.

        ---
        Built with [Streamlit](https://streamlit.io) · PyTorch · FastAPI backend
        """
    )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Predictive Maintenance",
        page_icon="🏭",
        layout="wide",
    )

    # ── Sidebar navigation ───────────────────────────────────────────────────
    with st.sidebar:
        st.title("🏭 PredMaint")
        st.caption("Industry 4.0 · Hydraulic Systems")
        st.divider()
        page = st.radio(
            "Navigation",
            options=["Predict", "Model Results", "About"],
            index=0,
        )
        st.divider()
        # Show model file status at a glance
        st.markdown("**Model files**")
        cnn_ok  = "✅" if CNN_MODEL_PATH.exists()     else "❌"
        siam_ok = "✅" if SIAMESE_MODEL_PATH.exists() else "❌"
        st.markdown(f"{cnn_ok} `cnn_model.pth`")
        st.markdown(f"{siam_ok} `siamese_model.pth`")

    # ── Route to the selected page ───────────────────────────────────────────
    if page == "Predict":
        page_predict()
    elif page == "Model Results":
        page_results()
    else:
        page_about()


if __name__ == "__main__":
    main()
