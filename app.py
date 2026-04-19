import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Traffic Flow Optimisation",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.8; }
    .signal-badge {
        display: inline-block;
        border-radius: 8px;
        padding: 4px 12px;
        font-weight: bold;
        font-size: 0.9rem;
        color: white;
    }
    .low      { background: #27ae60; }
    .moderate { background: #f39c12; }
    .high     { background: #e67e22; }
    .severe   { background: #c0392b; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_mat_data(file_bytes):
    import scipy.io as sio
    mat = sio.loadmat(io.BytesIO(file_bytes))
    # Try common variable names
    for key in mat:
        if not key.startswith("_"):
            val = mat[key]
            if hasattr(val, "toarray"):
                val = val.toarray()
            val = np.array(val, dtype=float)
            if val.ndim == 2 and val.shape[0] > 100:
                return val
    raise ValueError("Cannot parse .mat file. Please check dataset format.")


@st.cache_data(show_spinner=False)
def load_pems_data(file_bytes):
    """Parse PEMS-SF plain text file (space or semicolon separated floats)."""
    import re
    text = file_bytes.decode("utf-8")
    rows = []
    for line in text.strip().splitlines():
        line = line.strip().lstrip("[").rstrip("]")
        if not line:
            continue
        # Split on any combination of spaces, semicolons, commas
        tokens = re.split(r'[\s;,]+', line)
        vals = []
        for t in tokens:
            t = t.strip()
            if t:
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
        if vals:
            rows.append(vals)
    data = np.array(rows, dtype=float)
    if data.ndim == 2 and data.shape[1] > data.shape[0]:
        data = data.T
    return data


@st.cache_data(show_spinner=False)
def generate_synthetic_data(n_sensors=36, n_timesteps=2016, seed=42):
    """Fallback synthetic dataset mimicking the UCI traffic dataset statistics."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    base = np.sin(t) * 0.3 + 0.5
    data = np.zeros((n_timesteps, n_sensors))
    for s in range(n_sensors):
        phase = rng.uniform(0, np.pi)
        amp   = rng.uniform(0.2, 0.5)
        noise = rng.normal(0, 0.02, n_timesteps)
        data[:, s] = np.clip(amp * np.sin(t + phase) + 0.5 + noise, 0, 1)
    return data


def build_features(data, window=48):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i].flatten())
        y.append(data[i])
    return np.array(X), np.array(y)


@st.cache_resource(show_spinner=False)
def train_model(data_hash, _data, window, n_estimators, max_depth):
    X, y = build_features(_data, window)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42,
    )
    model = MultiOutputRegressor(rf)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²":   r2_score(y_test, y_pred),
    }
    return model, X_test, y_test, y_pred, metrics


def signal_recommendation(flow_value):
    if flow_value < 0.25:
        return "Low", "low", 30
    elif flow_value < 0.5:
        return "Moderate", "moderate", 45
    elif flow_value < 0.75:
        return "High", "high", 60
    else:
        return "Severe", "severe", 90


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-jam.png", width=80)
    st.title("⚙️ Configuration")
    st.markdown("---")

    st.subheader("📂 Dataset")
    uploaded = st.file_uploader(
        "Upload dataset (optional)",
        type=["mat", "txt", ""],
        help="Accepts: PEMS_train / PEMS_test (plain text) or a .mat file"
    )
    if not uploaded:
        st.info("No file uploaded — using synthetic dataset that mirrors UCI statistics.")

    st.markdown("---")
    st.subheader("🔧 Model Parameters")
    window       = st.slider("Input window (time steps)", 12, 96, 48, step=12)
    n_estimators = st.slider("RF estimators", 10, 200, 50, step=10)
    max_depth    = st.slider("Max depth", 3, 30, 10)

    st.markdown("---")
    st.subheader("📡 Sensor Selection")
    n_sensors_loaded = data.shape[1] if 'data' in dir() else 36
    sensor_idx = st.slider("Sensor to visualise", 0, max(n_sensors_loaded - 1, 35), 0)

    st.markdown("---")
    st.caption("**Author:** Nnamdi Onuigbo  \nAI Systems Engineer | SmartFlow Systems")


# ── Load data ─────────────────────────────────────────────────────────────────
if uploaded:
    try:
        fname = uploaded.name.lower()
        raw = uploaded.read()
        if fname.endswith(".mat"):
            data = load_mat_data(raw)
            data_source = f"UCI .mat — shape {data.shape}"
        else:
            # PEMS plain text file
            data = load_pems_data(raw)
            n_sensors = data.shape[1]
            data_source = f"PEMS dataset — {data.shape[0]} records × {n_sensors} sensors"
    except Exception as e:
        st.error(f"Could not load file: {e}")
        data = generate_synthetic_data()
        data_source = "Synthetic (fallback)"
else:
    data = generate_synthetic_data()
    data_source = "Synthetic (36 sensors × 2016 time steps)"

# Normalise to [0,1] if not already
if data.max() > 1.5:
    data = (data - data.min()) / (data.max() - data.min() + 1e-9)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚦 AI Traffic Flow Prediction & Signal Optimisation")
st.markdown(
    "Random Forest multi-output regression across **36 sensor locations**, "
    "with adaptive signal timing recommendations. | "
    f"Data: `{data_source}`"
)
st.markdown("---")

# ── Train ─────────────────────────────────────────────────────────────────────
with st.spinner("Training Random Forest model…"):
    data_hash = hash(data.tobytes())
    model, X_test, y_test, y_pred, metrics = train_model(
        data_hash, data, window, n_estimators, max_depth
    )

# ── Metric cards ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
kv = [
    ("MAE",  f"{metrics['MAE']:.4f}",  "Mean Absolute Error"),
    ("RMSE", f"{metrics['RMSE']:.4f}", "Root Mean Squared Error"),
    ("R²",   f"{metrics['R²']:.4f}",   "Variance Explained"),
    ("Sensors", "36", "Sensor Locations"),
]
for col, (title, val, sub) in zip([col1, col2, col3, col4], kv):
    col.markdown(
        f"""<div class="metric-card"><p>{title}</p><h2>{val}</h2><p>{sub}</p></div>""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Prediction vs Actual",
    "📊 Error Distribution",
    "🚦 Signal Recommendations",
    "🗺️ Sensor Heatmap",
])

# ── Tab 1: Prediction vs Actual ───────────────────────────────────────────────
with tab1:
    st.subheader(f"Sensor {sensor_idx} — Predicted vs Actual Traffic Flow")
    n_show = st.slider("Time steps to display", 50, min(500, len(y_test)), 150)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_test[:n_show, sensor_idx], label="Actual",    color="#2980b9", linewidth=1.5)
    ax.plot(y_pred[:n_show, sensor_idx], label="Predicted", color="#e74c3c",
            linewidth=1.5, linestyle="--", alpha=0.85)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Normalised Flow")
    ax.set_title(f"Sensor {sensor_idx} — Traffic Flow Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 2: Error Distribution ─────────────────────────────────────────────────
with tab2:
    st.subheader("Residual Error Distribution (All Sensors)")
    errors = (y_pred - y_test).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(errors, bins=80, color="#3498db", edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Prediction Error")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Histogram")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_test.flatten()[::10], y_pred.flatten()[::10],
                    alpha=0.3, s=4, color="#2ecc71")
    lims = [0, 1]
    axes[1].plot(lims, lims, "r--", linewidth=1.5)
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("Actual vs Predicted (scatter)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f"**Error stats** — Mean: {errors.mean():.5f} | "
        f"Std: {errors.std():.5f} | "
        f"Max absolute: {np.abs(errors).max():.5f}"
    )

# ── Tab 3: Signal Recommendations ─────────────────────────────────────────────
with tab3:
    st.subheader("🚦 Adaptive Signal Timing Recommendations")
    st.markdown(
        "Based on the **latest predicted traffic value** for each sensor, "
        "the system assigns a green-time recommendation using a threshold decision layer."
    )

    latest_pred = y_pred[-1]  # predictions for the most recent test step
    rows = []
    for i, flow in enumerate(latest_pred):
        level, badge, green = signal_recommendation(flow)
        rows.append({
            "Sensor": i,
            "Predicted Flow": round(float(flow), 4),
            "Traffic Level": level,
            "Green Time (s)": green,
        })

    df = pd.DataFrame(rows)

    # Colour-coded table
    def style_level(val):
        colours = {"Low": "#27ae60", "Moderate": "#f39c12",
                   "High": "#e67e22", "Severe": "#c0392b"}
        c = colours.get(val, "#666")
        return f"background-color: {c}; color: white; border-radius: 4px;"

    styled = df.style.map(style_level, subset=["Traffic Level"])
    st.dataframe(styled, use_container_width=True, height=500)

    # Summary bar
    summary = df["Traffic Level"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 3))
    colours_map = {"Low": "#27ae60", "Moderate": "#f39c12",
                   "High": "#e67e22", "Severe": "#c0392b"}
    bars = ax.bar(summary.index, summary.values,
                  color=[colours_map.get(l, "grey") for l in summary.index])
    ax.set_ylabel("Number of Sensors")
    ax.set_title("Traffic Level Distribution Across All Sensors")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2, str(int(bar.get_height())),
                ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Download
    csv = df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Signal Recommendations CSV",
        csv,
        "signal_recommendations.csv",
        "text/csv",
    )

# ── Tab 4: Sensor Heatmap ─────────────────────────────────────────────────────
with tab4:
    st.subheader("Sensor Traffic Heatmap — Last 100 Predicted Time Steps")
    n_steps = min(100, len(y_pred))
    heat_data = y_pred[-n_steps:].T   # shape: (36 sensors, n_steps)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(heat_data, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Normalised Traffic Flow")
    ax.set_xlabel("Time Step (last 100)")
    ax.set_ylabel("Sensor Index")
    ax.set_title("Traffic Flow Heatmap — All 36 Sensors")
    ax.set_yticks(range(0, 36, 3))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "🟢 Green = low flow   🟡 Yellow = moderate   🔴 Red = high / congested"
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:grey;font-size:0.8rem;'>"
    "AI Traffic Flow Prediction & Signal Optimisation · "
    "Nnamdi Onuigbo · SmartFlow Systems · "
    "<a href='https://github.com/onuigbonnamdi/ai-traffic-flow-optimisation' "
    "target='_blank'>GitHub</a>"
    "</center>",
    unsafe_allow_html=True,
)
