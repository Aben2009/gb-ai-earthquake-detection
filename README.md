# GB AI: Real-Time Earthquake Detection

A lightweight deep learning prototype that detects earthquakes and estimates their magnitude from a single seismic station, built for faster earthquake early warning in Indonesia.

🥈 **Silver medal, World Youth Invention and Innovation Award (WYIIA) 2026**

## What it does

GB AI reads a 30-second, 3-component seismic waveform and re-checks it every 0.1–0.2 seconds (sliding window). For every window it outputs:

- **Detection**: earthquake or noise
- **Magnitude**: estimated size of the earthquake
- **Uncertainty**: how confident the model is about that magnitude

The repository includes a live dashboard that replays real earthquake recordings as a data stream and raises an alert when GB AI detects the P-wave.

## Model

| Part | Configuration |
| --- | --- |
| Feature extraction | Conv1D 32 → 64 → 128 filters, BatchNorm, ReLU, MaxPool |
| Sequence analysis | Bidirectional LSTM (256 units), dropout 0.5, additive attention |
| Outputs | Detection, magnitude, uncertainty (multi-task) |
| Size | 858,884 trainable parameters |

## Results (prototype, ETHZ benchmark)

| Metric | GB AI | Baseline CNN |
| --- | --- | --- |
| Detection accuracy | 98.1% | 95.6% |
| F1 score | 98.1% | 95.5% |
| False-alarm rate (per window) | 1.4% | 2.4% |
| Magnitude MAE (all events) | 0.35 | – |

Replayed real-time test on 200 recordings: 99.5% detected, median **0.20 s** of P-wave data needed to raise an alert, 4.5% of recordings had an early false alarm. Inference takes about **63 ms per window** on a free Google Colab CPU when the model is compiled into a static graph (see `GB_AI_Evaluation.ipynb`); the original paper measured 1.6 s per window with step-by-step (eager) execution.

## Limitations

- Trained on Swiss data (ETHZ), not yet on Indonesian data.
- Magnitude is underestimated above M3, because the training data has few large earthquakes.
- Trained for only 11 epochs on a CPU.
- Noise examples are quiet pre-event periods; landslides, volcanic tremor and traffic were not included.

## Run the live demo

```bash
git clone https://github.com/Aben2009/gb-ai-earthquake-detection.git
cd gb-ai-earthquake-detection
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Then pick a recording in the sidebar and press **Start live demo**.

## Train it yourself

Open `GB_AI_Prototype_Colab.ipynb` in Google Colab (GPU runtime recommended). It downloads the ETHZ dataset through SeisBench, trains the baseline CNN and GB AI, evaluates them, and saves `gb_ai.keras`. Run `make_demo_data.py` afterwards to create the small demo file used by the dashboard.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Live Streamlit dashboard |
| `GB_AI_Prototype_Colab.ipynb` | Data preparation, training and evaluation (code; the original training run's outputs were not saved) |
| `GB_AI_Evaluation.ipynb` | Re-runs the evaluation of the saved model, with outputs |
| `hasil_prototype.csv` | Summary of the original prototype results from Colab |
| `gb_ai.keras` | Trained GB AI model |
| `gbai_demo_recordings.npz` | Demo recordings replayed by the dashboard |
| `make_demo_data.py` | Extracts the demo recordings from the full Colab cache |

## Team

SMAIT As-Syifa Boarding School Wanareja: Ahmad Benjamin Wahyudi (team leader), Ariq Zahran El Shirezy, Muhammad Anwar Fathurrohman, Rizq Kasyfi Hidayat, Shaka Aqilla Al-Faqih, Tegar Nurfauzan.

## Acknowledgements

- ETHZ dataset from the Swiss Seismological Service (SED) at ETH Zurich, accessed through [SeisBench](https://github.com/seisbench/seisbench) (Woollam et al., 2022).
- Inspired by PhaseNet (Zhu & Beroza, 2019) and Earthquake Transformer (Mousavi et al., 2020).

This is a research prototype, not an operational warning system.
