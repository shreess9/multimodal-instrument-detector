\
# 🎵 Multimodal Instrument Detector 🎥

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/pytorch-%5E2.0-red)](https://pytorch.org/)

A **Python-based AI system** to automatically detect if a person is playing a musical instrument in a video using **vision + audio multimodal analysis**. The system fuses visual cues from frames with audio features to provide a **fused probability** and a clear decision.

---

## 🌟 Features

- **Vision Analysis**
  - Samples multiple frames from the video.
  - Uses **pretrained ResNet-18** on ImageNet for instrument detection.
  - Lists **top visual instrument cues** per video.

- **Audio Analysis**
  - Extracts audio via **FFmpeg** or **MoviePy**.
  - Computes musical likelihood using:
    - Energy & silence ratio
    - Harmonic vs percussive content
    - Chroma richness
    - Tempo and spectral centroid

- **Multimodal Fusion**
  - Weighted combination of vision and audio likelihood.
  - Provides final **fused confidence score**.

- **Interactive Tkinter UI**
  - Visualizes per-frame instrument probability, audio likelihood, and fused score.
  - Displays top instruments in a neat table.
  - Save JSON/PNG diagnostics directly from the UI.

- **CLI & Diagnostics**
  - Command-line interface for headless use.
  - Optional **diagnostic plots and JSON output**.

---

## 🖥️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/multimodal-instrument-detector.git
cd multimodal-instrument-detector
````

2. **Set up Python environment**

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

**Dependencies included:**

* `torch`, `torchvision`
* `numpy`, `scipy`, `matplotlib`, `Pillow`
* `opencv-python`
* `librosa`, `soundfile`
* `moviepy`, `imageio-ffmpeg`
* `tkinter` (preinstalled with Python in most distributions)

---

## 🚀 Usage

### Command-Line Interface

```bash
python main.py path/to/video.mp4 --threshold 0.5 --diagnostics
```

**Options:**

* `video` : Path to the video file (must have audio).
* `--threshold` : Decision threshold for fused score (default: 0.5).
* `--diagnostics` : Save PNG/JSON diagnostics.
* `--no-ui` : Run CLI only, skip Tkinter window.

---

### Example CLI Output

```
==== RESULT ====
Decision          : Instrument playing
Fused confidence  : 0.78
Vision summary    : 0.72
Audio likelihood  : 0.65
=================
Top visual instrument cues:
  - acoustic guitar: 0.81
  - piano: 0.45
```

---

### UI Example

* Interactive window showing:

  * **Per-frame instrument probabilities**
  * **Audio music-likelihood**
  * **Fused probability**
  * **Top visual instruments table**
* Buttons to save diagnostics as **JSON** and **PNG**

---

## 🔍 How it Works

1. **Frame Sampling**

   * Uniformly sample `NUM_FRAMES` frames from video.

2. **Vision Branch**

   * Preprocess frames → pass through **ResNet-18**.
   * Aggregate instrument-related label probabilities.

3. **Audio Branch**

   * Extract mono WAV at `TARGET_SR`.
   * Compute music-likelihood using energy, silence, harmonicity, chroma, tempo, spectral centroid.

4. **Fusion**

   * Weighted combination of vision and audio likelihood:
     `fused = 1 - ((1-vision_prob)^vision_weight * (1-audio_prob)^audio_weight)`

5. **Decision**

   * Fused score ≥ threshold → **Instrument playing**, else → **Not instrument playing**

---

## 📂 Output Files

* **CLI**: Prints summary and top instruments.
* **UI**: Displays interactive visualization.
* **Diagnostics**:

  * `video_diagnostics.png` → per-frame & fused probabilities plot
  * `video_diagnostics.json` → detailed scores & top instruments

---

## ⚙️ Notes

* Works best with clear visual and audio cues.
* Audio extraction requires **FFmpeg** installed.
* Can handle videos without audio (falls back to vision-only detection).

---

## 🔮 Future Improvements

* Add **video transformers** or **temporal CNNs** for improved detection.
* Real-time webcam/streaming support.
* Handle **overlapping instruments** in polyphonic audio.
* Improved UI with live audio waveform visualization.

---

## 📄 License

MIT License © [Your Name]

---

## 📚 References

* [PyTorch Pretrained Models](https://pytorch.org/vision/stable/models.html)
* [Librosa: Audio Analysis](https://librosa.org/)
* [MoviePy Video Processing](https://zulko.github.io/moviepy/)
* [Tkinter GUI](https://docs.python.org/3/library/tkinter.html)


