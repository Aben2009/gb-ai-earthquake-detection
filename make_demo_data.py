"""Extract only the demo recordings from the full Colab cache.

The full cache (gbai_windows_30s.npz) holds every training window and is far
larger than GitHub's 100 MB file limit. The live demo only needs demo_X and demo_m.

Usage:  python make_demo_data.py
"""
import numpy as np

src = np.load("gbai_windows_30s.npz")
np.savez_compressed("gbai_demo_recordings.npz",
                    demo_X=src["demo_X"].astype(np.float32),
                    demo_m=src["demo_m"].astype(np.float32))
print("Saved gbai_demo_recordings.npz with", len(src["demo_X"]), "recordings")
