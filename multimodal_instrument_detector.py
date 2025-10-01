import argparse
import os
import sys
import math
import _tkinter
import json
import tempfile
from dataclasses import dataclass
import subprocess, shlex, tempfile, os
from imageio_ffmpeg import get_ffmpeg_exe

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from moviepy.editor import VideoFileClip
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import subprocess, shlex, tempfile, os
from imageio_ffmpeg import get_ffmpeg_exe
from dataclasses import dataclass
from typing import List, Tuple

from collections import defaultdict
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -------------------------
# Config
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Keywords to match ImageNet labels that imply instruments
INSTRUMENT_KEYWORDS = [
    "guitar", "acoustic", "electric guitar", "banjo", "cello", "violin", "fiddle",
    "viola", "double bass", "piano", "grand piano", "keyboard", "organ", "accordion",
    "trumpet", "trombone", "tuba", "sax", "saxophone", "clarinet", "flute",
    "harmonica", "drum", "drums", "snare", "timpani", "marimba", "xylophone",
    "tambourine", "balalaika", "bagpipe", "harp", "ukulele", "oboe", "bassoon"
]

# Vision parameters
NUM_FRAMES = 16            # sample this many frames uniformly from the video
MIN_SIDE = 256             # resize short side to this before center-crop
CROP_SIZE = 224            # model input

# Audio parameters
TARGET_SR = 16000
HOP_LENGTH = 512
N_FFT = 2048

# Fusion weight (you can tweak)
VISION_WEIGHT = 0.55
AUDIO_WEIGHT = 0.45

@dataclass
class AudioScores:
    music_likelihood: float
    diagnostics: dict

@dataclass
class VisionScores:
    per_frame_probs: list
    summary_prob: float
    top_instruments: List[Tuple[str, float]]  # (label, avg_prob)


# -------------------------
# Utils
# -------------------------
def safe_print(*args, **kwargs):
    """Print that flushes immediately (nicer in terminals)."""
    print(*args, **kwargs, flush=True)


def load_imagenet_categories_from_weights(weights):
    # Torchvision weights have meta with category names
    categories = weights.meta.get("categories", None)
    if categories is None:
        raise RuntimeError("Could not find categories in weights meta.")
    return categories


def has_instrument_keyword(label: str) -> bool:
    lbl = label.lower()
    for kw in INSTRUMENT_KEYWORDS:
        if kw in lbl:
            return True
    return False


def build_vision_model():
    # Use ResNet18 pretrained on ImageNet
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval().to(DEVICE)

    # Use the official preprocessing pipeline (includes Resize, Crop, Normalize)
    preprocess = weights.transforms()

    # Category labels
    categories = weights.meta.get("categories", None)
    if categories is None:
        raise RuntimeError("Could not load ImageNet categories from weights.")

    return model, preprocess, categories


def sample_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, frame_count - 1), num_frames).astype(int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def infer_vision_instrument_prob(model, preprocess, categories, frames, topk=5):
    if len(frames) == 0:
        return VisionScores(per_frame_probs=[], summary_prob=0.0, top_instruments=[])

    per_frame_probs = []
    label_scores = defaultdict(list)  # label_name -> [per-frame probs]

    with torch.no_grad():
        for f in frames:
            # f can be a PIL.Image already if you changed sample_frames as suggested; if it's np.ndarray, wrap it:
            if isinstance(f, np.ndarray):
                pil_img = Image.fromarray(f)
            else:
                pil_img = f

            inp = preprocess(pil_img).unsqueeze(0).to(DEVICE)
            logits = model(inp)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # shape [1000]

            # Sum over instrument-like labels for the overall instrument probability
            instrument_prob = 0.0
            for idx, p in enumerate(probs):
                lbl = categories[idx]
                if has_instrument_keyword(lbl):
                    instrument_prob += float(p)
                    label_scores[lbl].append(float(p))

            instrument_prob = float(min(1.0, instrument_prob))
            per_frame_probs.append(instrument_prob)

    # Robust aggregate: mean of top half of per-frame instrument probs
    top_half = sorted(per_frame_probs, reverse=True)[: max(1, len(per_frame_probs)//2)]
    summary = float(np.mean(top_half)) if top_half else 0.0

    # Compute average per-label and pick top-k instruments
    avg_label_scores = [(lbl, float(np.mean(scores))) for lbl, scores in label_scores.items() if scores]
    avg_label_scores.sort(key=lambda x: x[1], reverse=True)
    top_instruments = avg_label_scores[:topk]

    return VisionScores(per_frame_probs=per_frame_probs, summary_prob=summary, top_instruments=top_instruments)


def extract_audio_to_wav(video_path, target_sr=16000):
    """
    1) Probe for audio stream.
    2) Try FFmpeg demux to mono WAV at target_sr.
    3) Fallback to MoviePy write_audiofile if FFmpeg demux fails.
    Returns a temp .wav path or raises RuntimeError if both methods fail.
    """
    probe = probe_audio_stream(video_path)
    if not probe["has_audio"]:
        raise RuntimeError("No audio stream found in the input video (probe).")

    # --- Try FFmpeg path ---
    ffmpeg = get_ffmpeg_exe()
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav_path = tmp_wav.name
    tmp_wav.close()

    cmd = f'"{ffmpeg}" -hide_banner -loglevel error -i "{video_path}" -vn -ac 1 -ar {target_sr} -f wav -y "{tmp_wav_path}"'
    proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 0 and os.path.exists(tmp_wav_path) and os.path.getsize(tmp_wav_path) > 0:
        return tmp_wav_path

    # --- Fallback: MoviePy write_audiofile (uses ffmpeg under the hood) ---
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            raise RuntimeError("MoviePy reports no audio track.")
        # Ensure mono + sr
        clip.audio.write_audiofile(tmp_wav_path, fps=target_sr, nbytes=2, codec="pcm_s16le", verbose=False, logger=None)
        clip.close()
        if os.path.exists(tmp_wav_path) and os.path.getsize(tmp_wav_path) > 0:
            return tmp_wav_path
    except Exception as e:
        pass

    # If both failed:
    err = proc.stderr.decode(errors="ignore")
    raise RuntimeError(f"Audio extraction failed via FFmpeg & MoviePy.\nFFmpeg error:\n{err}")

def compute_audio_music_likelihood(wav_path, sr=TARGET_SR):
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    if len(y) == 0:
        return AudioScores(music_likelihood=0.0, diagnostics={})

    # Basic energy & silence ratio
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    energy = float(np.mean(rms))
    silence_ratio = float(np.mean(rms < (np.percentile(rms, 25))))

    # Harmonic-percussive separation: music tends to have stronger harmonic component
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_ratio = float((np.linalg.norm(y_harm) + 1e-8) / (np.linalg.norm(y) + 1e-8))

    # Chroma richness: tonal content across pitch classes
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_spread = float(np.mean(np.std(chroma, axis=1)))  # variability across classes

    # Spectral centroid (music often in a mid-band vs speech noise)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_norm = float(np.clip(np.mean(centroid) / (sr/2), 0, 1))

    # Tempo presence (perceived steady beat)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_norm = float(np.clip(tempo / 200.0, 0, 1))  # crude scaling

    # Heuristic scoring in [0,1]
    # - More energy → more likely
    # - Lower silence ratio → more likely
    # - Higher harmonicity, chroma spread, and some tempo → more likely
    # - Centroid mid-range (~0.2–0.6) preferred: penalize extremes
    centroid_mid_pref = 1.0 - abs(centroid_norm - 0.4) / 0.4
    centroid_mid_pref = float(np.clip(centroid_mid_pref, 0, 1))

    # Normalize energy roughly (robust to scale)
    energy_norm = float(np.clip((energy - 0.005) / 0.02, 0, 1))
    silence_inv = 1.0 - silence_ratio

    components = {
        "energy": energy_norm,
        "silence_inv": silence_inv,
        "harmonicity": float(np.clip(harm_ratio * 1.5, 0, 1)),
        "chroma_spread": float(np.clip(chroma_spread / 0.25, 0, 1)),
        "tempo": tempo_norm,
        "centroid_mid_pref": centroid_mid_pref
    }

    # Weighted average (tweakable)
    weights = {
        "energy": 0.20,
        "silence_inv": 0.15,
        "harmonicity": 0.25,
        "chroma_spread": 0.20,
        "tempo": 0.10,
        "centroid_mid_pref": 0.10
    }

    score = 0.0
    for k, w in weights.items():
        score += components[k] * w

    score = float(np.clip(score, 0, 1))
    return AudioScores(music_likelihood=score, diagnostics=components)


def fuse_scores(vision_prob, audio_prob, vw=VISION_WEIGHT, aw=AUDIO_WEIGHT):
    # Weighted noisy-OR: 1 - Π(1 - p_i^w_i)
    v_term = (1.0 - vision_prob) ** vw
    a_term = (1.0 - audio_prob) ** aw
    fused = 1.0 - (v_term * a_term)
    return float(np.clip(fused, 0, 1))


def save_diagnostics_plot(out_path, frame_probs, vision_total, audio_score, fused_score):
    plt.figure(figsize=(8, 5))
    x = np.arange(len(frame_probs))
    plt.plot(x, frame_probs, marker='o', label='Vision: per-frame instrument prob')
    plt.axhline(vision_total, linestyle='--', label=f'Vision summary={vision_total:.2f}')
    plt.axhline(audio_score, linestyle='-.', label=f'Audio music-likelihood={audio_score:.2f}')
    plt.axhline(fused_score, linestyle=':', label=f'Fused={fused_score:.2f}')
    plt.ylim(0, 1)
    plt.xlabel('Sampled frame index')
    plt.ylabel('Probability / Likelihood')
    plt.title('Multimodal Instrument-Playing Diagnostics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def build_diagnostics_figure(frame_probs, vision_total, audio_score, fused_score):
    """Return a matplotlib Figure visualizing the diagnostics (no file I/O)."""
    fig = plt.figure(figsize=(7, 4.5))
    x = np.arange(len(frame_probs))
    plt.plot(x, frame_probs, marker='o', label='Vision per-frame')
    plt.axhline(vision_total, linestyle='--', label=f'Vision summary={vision_total:.2f}')
    plt.axhline(audio_score, linestyle='-.', label=f'Audio likelihood={audio_score:.2f}')
    plt.axhline(fused_score, linestyle=':', label=f'Fused={fused_score:.2f}')
    plt.ylim(0, 1)
    plt.xlabel('Sampled frame index')
    plt.ylabel('Probability / Likelihood')
    plt.title('Multimodal Instrument-Playing Diagnostics')
    plt.legend(loc='best')
    plt.tight_layout()
    return fig


def show_ui(video_path, v_scores, a_scores, fused, decision):
    """Simple Tkinter UI to display the diagnostics."""
    root = tk.Tk()
    root.title("Multimodal Instrument Detector — Diagnostics")

    # --- Top section: headline decision ---
    headline = ttk.Label(
        root,
        text=f"Video: {os.path.basename(video_path)}\nDecision: {decision}   |   Fused: {fused:.3f}",
        font=("Segoe UI", 12, "bold"),
        anchor="center",
        justify="center"
    )
    headline.grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 6), sticky="ew")

    # --- Left panel: key scores + top instruments ---
    left_frame = ttk.Frame(root, padding=8)
    left_frame.grid(row=1, column=0, sticky="nsew")

    ttk.Label(left_frame, text=f"Vision summary: {v_scores.summary_prob:.3f}", font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 4))
    ttk.Label(left_frame, text=f"Audio likelihood: {a_scores.music_likelihood:.3f}", font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 8))

    ttk.Label(left_frame, text="Top visual instrument cues", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 4))

    # Table for top instruments
    tree = ttk.Treeview(left_frame, columns=("label", "score"), show="headings", height=8)
    tree.heading("label", text="Label")
    tree.heading("score", text="Avg Prob")
    tree.column("label", width=180, anchor="w")
    tree.column("score", width=80, anchor="center")
    tree.pack(fill="both", expand=True)

    if getattr(v_scores, "top_instruments", None):
        for lbl, s in v_scores.top_instruments:
            tree.insert("", "end", values=(lbl, f"{s:.3f}"))
    else:
        tree.insert("", "end", values=("—", "—"))

    # --- Right panel: matplotlib figure (per-frame + lines) ---
    right_frame = ttk.Frame(root, padding=8)
    right_frame.grid(row=1, column=1, sticky="nsew")

    fig = build_diagnostics_figure(
        v_scores.per_frame_probs,
        v_scores.summary_prob,
        a_scores.music_likelihood,
        fused
    )
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- Bottom bar: Save buttons (JSON + PNG) ---
    bottom = ttk.Frame(root, padding=8)
    bottom.grid(row=2, column=0, columnspan=2, sticky="ew")

    status_var = tk.StringVar(value="Ready")
    status_label = ttk.Label(bottom, textvariable=status_var)
    status_label.pack(side="left")

    def save_json():
        # reuse your existing json structure if you want; here we reconstruct lightweightly
        out = {
            "vision": {
                "per_frame_probs": [round(p, 4) for p in v_scores.per_frame_probs],
                "summary_prob": round(v_scores.summary_prob, 4),
                "top_instruments": [(lbl, round(s, 4)) for (lbl, s) in getattr(v_scores, "top_instruments", [])],
            },
            "audio": {
                "music_likelihood": round(a_scores.music_likelihood, 4),
                "components": {k: round(v, 4) for k, v in a_scores.diagnostics.items()},
            },
            "fused": round(fused, 4),
            "decision": decision,
        }
        path = os.path.splitext(os.path.basename(video_path))[0] + "_ui_diagnostics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        status_var.set(f"Saved {path}")

    def save_png():
        path = os.path.splitext(os.path.basename(video_path))[0] + "_ui_plot.png"
        fig.savefig(path, dpi=180)
        status_var.set(f"Saved {path}")

    ttk.Button(bottom, text="Save JSON", command=save_json).pack(side="right", padx=(6, 0))
    ttk.Button(bottom, text="Save Plot PNG", command=save_png).pack(side="right", padx=(6, 0))

    # --- window sizing behaviour ---
    root.grid_columnconfigure(0, weight=1, uniform="col")
    root.grid_columnconfigure(1, weight=1, uniform="col")
    root.grid_rowconfigure(1, weight=1)

    root.mainloop()

def probe_audio_stream(video_path) -> dict:
    """
    Return {'has_audio': bool, 'codec': str|None, 'stderr': str} by probing with ffmpeg.
    """
    ffmpeg = get_ffmpeg_exe()
    cmd = f'"{ffmpeg}" -hide_banner -i "{video_path}" -f null -'
    proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = proc.stderr.decode(errors="ignore")
    has_audio = ("Audio:" in stderr)
    codec = None
    if has_audio:
        # Grab first "Audio: <codec>"
        for line in stderr.splitlines():
            if "Audio:" in line:
                # Example: Stream #0:1(und): Audio: aac (LC), 48000 Hz, stereo, fltp, 125 kb/s
                parts = line.split("Audio:")
                if len(parts) > 1:
                    codec = parts[1].strip().split(",")[0].strip()
                break
    return {"has_audio": has_audio, "codec": codec, "stderr": stderr}


def main():
    parser = argparse.ArgumentParser(description="Multimodal Instrument Playing Detector")
    parser.add_argument("video", help="Path to a video file (with audio)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on fused score")
    parser.add_argument("--diagnostics", action="store_true", help="Also save a diagnostics PNG/JSON to disk")
    parser.add_argument("--no-ui", action="store_true", help="Skip the UI window (CLI only)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        safe_print(f"ERROR: File not found: {args.video}")
        sys.exit(1)

    safe_print(f"Using device: {DEVICE}")

    # Vision
    safe_print("Loading vision model...")
    model, preprocess, categories = build_vision_model()

    safe_print("Sampling frames...")
    frames = sample_frames(args.video, NUM_FRAMES)
    safe_print(f"Sampled {len(frames)} frames.")

    safe_print("Running vision inference...")
    v_scores = infer_vision_instrument_prob(model, preprocess, categories, frames)
    safe_print(f"Vision instrument probability (summary): {v_scores.summary_prob:.3f}")

    # Audio
    safe_print("Extracting audio...")
    a_scores = None
    tmp_wav = None
    try:
        tmp_wav = extract_audio_to_wav(args.video, TARGET_SR)
        safe_print("Analyzing audio features...")
        a_scores = compute_audio_music_likelihood(tmp_wav, TARGET_SR)
    except Exception as e:
        safe_print(f"[WARN] Audio extraction/analysis failed: {e}")
        # proceed with zeroed audio (vision-only fallback)
        a_scores = AudioScores(music_likelihood=0.0, diagnostics={"note": "audio_unavailable_or_extraction_failed"})
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass
    safe_print(f"Audio music-likelihood: {a_scores.music_likelihood:.3f}")

    # Fusion
    fused = fuse_scores(v_scores.summary_prob, a_scores.music_likelihood, VISION_WEIGHT, AUDIO_WEIGHT)
    decision = "Instrument playing" if fused >= args.threshold else "Not instrument playing"

    # --- CLI summary ---
    safe_print("\n==== RESULT ====")
    safe_print(f"Decision          : {decision}")
    safe_print(f"Fused confidence  : {fused:.3f}")
    safe_print(f"Vision summary    : {v_scores.summary_prob:.3f}")
    safe_print(f"Audio likelihood  : {a_scores.music_likelihood:.3f}")
    safe_print("=================\n")

    if getattr(v_scores, "top_instruments", None):
        safe_print("Top visual instrument cues:")
        for lbl, s in v_scores.top_instruments:
            safe_print(f"  - {lbl}: {s:.3f}")
    else:
        safe_print("No specific instrument label stood out in the vision branch.")

    # --- Optional file diagnostics (same as before) ---
    if args.diagnostics:
        out_stem = os.path.splitext(os.path.basename(args.video))[0]
        save_diagnostics_plot(
            out_stem + "_diagnostics.png",
            v_scores.per_frame_probs,
            v_scores.summary_prob,
            a_scores.music_likelihood,
            fused
        )
        diag_json = {
            "vision": {
                "per_frame_probs": [round(p, 4) for p in v_scores.per_frame_probs],
                "summary_prob": round(v_scores.summary_prob, 4),
                "top_instruments": [(lbl, round(s, 4)) for (lbl, s) in getattr(v_scores, "top_instruments", [])],
            },
            "audio": {
                "music_likelihood": round(a_scores.music_likelihood, 4),
                "components": {k: round(v, 4) for k, v in a_scores.diagnostics.items()},
            },
            "fused": round(fused, 4),
            "threshold": args.threshold,
            "decision": decision
        }
        with open(out_stem + "_diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(diag_json, f, indent=2)
        safe_print(f"Saved diagnostics files → {out_stem}_diagnostics.(png|json)")

    # --- UI window (default on; disable with --no-ui) ---
    if not args.no_ui:
        show_ui(args.video, v_scores, a_scores, fused, decision)


if __name__ == "__main__":
    main()
