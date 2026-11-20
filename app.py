"""
GB AI - Live Earthquake Detection Demo (software prototype)
Run:  streamlit run app.py
Needs in the same folder: gb_ai.keras and gbai_demo_recordings.npz (see README)
"""
import os
import time
import json
import urllib.parse
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras

SR = 100                      # samples per second
POST = 15 * SR                # data after the P-wave stored in each demo recording
MODEL_FILE = "gb_ai.keras"
# small demo file (made by make_demo_data.py); falls back to the full Colab cache
DATA_FILE = "gbai_demo_recordings.npz" if os.path.exists("gbai_demo_recordings.npz") else "gbai_windows_30s.npz"

st.set_page_config(page_title="GB AI Live Demo", page_icon="🌏", layout="wide")


# ---------------------------------------------------------------- loading
def mag_loss(y_true, y_pred):  # needed only so the saved model can be read
    m, mask = y_true[:, 0], y_true[:, 1]
    mu, logvar = y_pred[:, 0], tf.clip_by_value(y_pred[:, 1], -6.0, 6.0)
    return mask * (0.5 * tf.abs(m - mu) + 0.3 * 0.5 * (tf.exp(-logvar) * tf.square(m - mu) + logvar))


@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_FILE, custom_objects={"mag_loss": mag_loss}, compile=False)
    win = int(model.input_shape[1])

    # Compile the model into a static graph once: much faster than calling it eagerly every update.
    @tf.function(input_signature=[tf.TensorSpec((1, win, 3), tf.float32)])
    def predict(x):
        return model(x, training=False)

    for _ in range(3):  # warm-up (first calls build the graph)
        predict(tf.zeros((1, win, 3)))
    return predict, win


@st.cache_data
def load_recordings():
    d = np.load(DATA_FILE)
    return d["demo_X"].astype(np.float32), d["demo_m"].astype(np.float32)


def normalize(w):
    """(3, WIN) -> (WIN, 3), same z-score normalization as in training."""
    w = w - w.mean(axis=1, keepdims=True)
    w = w / (w.std(axis=1, keepdims=True) + 1e-8)
    return w.T.astype(np.float32)


def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=5) as r:
            return json.loads(r.read().decode()).get("ok", False)
    except Exception:
        return False


# ---------------------------------------------------------------- checks
missing = [f for f in (MODEL_FILE, DATA_FILE) if not os.path.exists(f)]
if missing:
    st.error("File not found: " + ", ".join(missing) +
             ". Put them in the same folder as app.py (see README).")
    st.stop()

with st.spinner("Loading GB AI model..."):
    model, WIN = load_model()
recs, mags = load_recordings()
P_IDX = recs.shape[2] - POST          # position of the P-wave in every demo recording
PRE_SEC = (P_IDX - WIN) / SR          # seconds of data available before the P-wave

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Demo settings")
rec_id = st.sidebar.number_input("Recording #", 0, len(recs) - 1, 0, 1)
start = st.sidebar.slider("Start (s before P-wave)", -float(PRE_SEC), 0.0, -5.0, 0.5)
end = st.sidebar.slider("Stop (s after P-wave)", 1.0, POST / SR, 10.0, 0.5)
step = st.sidebar.select_slider("Update every (s)", [0.1, 0.2, 0.5], value=0.2)
thr = st.sidebar.slider("Alert threshold", 0.1, 0.9, 0.5, 0.05)
show_p = st.sidebar.checkbox("Show catalog P-wave time (reference)", True)
st.sidebar.markdown("---")
st.sidebar.subheader("Phone alert (optional)")
tg_token = st.sidebar.text_input("Telegram bot token", type="password")
tg_chat = st.sidebar.text_input("Telegram chat ID")
run = st.sidebar.button("▶ Start live demo", type="primary", use_container_width=True)

# ---------------------------------------------------------------- layout
st.title("GB AI: Live Earthquake Detection")
st.caption("Software prototype. Real earthquake recordings (ETHZ benchmark) are replayed as a live "
           "data stream; GB AI analyses a 30-second window at every update.")

status = st.empty()
c1, c2, c3, c4 = st.columns(4)
m_time, m_prob, m_mag, m_inf = c1.empty(), c2.empty(), c3.empty(), c4.empty()
chart = st.empty()
summary = st.empty()


def show_status(alert, early=False):
    if alert:
        msg = "ALERT: earthquake detected" + (" (before catalog P-wave, false alarm)" if early else "")
        status.markdown(f"<div style='background:#E24B4A;color:white;padding:14px;border-radius:10px;"
                        f"font-size:26px;font-weight:700;text-align:center'>{msg}</div>",
                        unsafe_allow_html=True)
    else:
        status.markdown("<div style='background:#1D9E75;color:white;padding:14px;border-radius:10px;"
                        "font-size:26px;font-weight:700;text-align:center'>Monitoring: no earthquake</div>",
                        unsafe_allow_html=True)


def draw(seg, e, times, probs, alert_t):
    t_now = (e - P_IDX) / SR
    t_win = (np.arange(e - WIN, e) - P_IDX) / SR
    plt.style.use("dark_background")
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.2), gridspec_kw={"height_ratios": [1.3, 1]})
    ax[0].plot(t_win, seg[0, e - WIN:e], lw=0.6, color="white")
    ax[0].set_xlim(t_win[0], t_win[-1])
    ax[0].set_ylabel("Ground motion (Z)")
    ax[0].set_title("Incoming seismic data: last 30 s seen by GB AI", fontsize=11)
    ax[1].plot(times, probs, color="#EF9F27", lw=2, label="Earthquake probability")
    ax[1].axhline(thr, color="gray", ls=":", label=f"Threshold {thr:.2f}")
    ax[1].set_xlim(start, end)
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_xlabel("Seconds relative to catalog P-wave arrival")
    for a in ax:
        if show_p and t_now >= 0:
            a.axvline(0, color="#378ADD", ls="--", lw=1.2, label="P-wave (catalog)" if a is ax[1] else None)
        if alert_t is not None:
            a.axvline(alert_t, color="#E24B4A", lw=2, label=f"ALERT at {alert_t:+.1f} s" if a is ax[1] else None)
    ax[1].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    chart.pyplot(fig)
    plt.close(fig)


show_status(False)
if not run:
    st.info("Choose a recording in the sidebar and press **Start live demo**.")
    st.stop()

# ---------------------------------------------------------------- live loop
seg = recs[int(rec_id)]
ends = np.arange(P_IDX + int(start * SR), P_IDX + int(end * SR) + 1, int(step * SR))
times, probs, inf_ms = [], [], []
alert_t, alert_mag, sent = None, None, False

for e in ends:
    t0 = time.perf_counter()
    x = normalize(seg[:, e - WIN:e])[None]
    det, mag = model(tf.constant(x))
    det, mag = det.numpy(), mag.numpy()   # wait for the result before stopping the clock
    ms = (time.perf_counter() - t0) * 1000
    p = float(det[0, 0])
    mu = float(mag[0, 0])
    sigma = float(np.sqrt(np.exp(np.clip(mag[0, 1], -6, 6))))
    t = (e - P_IDX) / SR
    times.append(t); probs.append(p); inf_ms.append(ms)

    if alert_t is None and p >= thr:
        alert_t, alert_mag = t, (mu, sigma)
        if tg_token and tg_chat and not sent:
            sent = send_telegram(tg_token, tg_chat,
                                 f"GB AI EARTHQUAKE ALERT\nEstimated magnitude: M {mu:.1f} ± {sigma:.1f}\n"
                                 f"Take cover now. (Demo: replayed recording #{int(rec_id)})")

    show_status(alert_t is not None, early=alert_t is not None and alert_t < 0)
    m_time.metric("Stream time (vs P-wave)", f"{t:+.1f} s")
    m_prob.metric("Earthquake probability", f"{p:.2f}")
    m_mag.metric("Magnitude estimate", f"M {mu:.1f} ± {sigma:.1f}" if p >= thr else "-")
    m_inf.metric("Inference time", f"{ms:.0f} ms")
    draw(seg, e, times, probs, alert_t)

    lag = step - (time.perf_counter() - t0)   # pace the replay to real time when possible
    if lag > 0:
        time.sleep(lag)

# ---------------------------------------------------------------- summary
lines = [f"**Recording #{int(rec_id)}**, catalog magnitude **{mags[int(rec_id)]:.1f}**"]
if alert_t is None:
    lines.append("No alert raised in this time span.")
elif alert_t < 0:
    lines.append(f"Alert raised **{-alert_t:.1f} s before** the catalog P-wave: a false alarm on noise.")
else:
    lines.append(f"Alert raised **{alert_t:.1f} s after** the P-wave arrived "
                 f"(estimated M {alert_mag[0]:.1f} ± {alert_mag[1]:.1f}).")
lines.append(f"Average inference time on this computer: **{np.mean(inf_ms):.0f} ms** per window "
             f"({len(inf_ms)} windows).")
if np.mean(inf_ms) > step * 1000:
    lines.append("Note: inference was slower than the update interval, so the replay ran slower than real time.")
if tg_token and tg_chat:
    lines.append("Phone alert: " + ("sent ✅" if sent else "not sent (check token / chat ID / internet)"))
summary.success("  \n".join(lines))
