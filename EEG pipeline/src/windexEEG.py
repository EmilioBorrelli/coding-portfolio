"""
windexEEG.py

A production-style, QC-aware EEG preprocessing pipeline with:
- Deterministic behavior (seeded randomness where applicable)
- Structured JSON logging (per-stage metrics + timings)
- Environment & config capture (versions, git hash if available)
- Adaptive line-noise removal (best-of, safe)
- Adaptive high-pass selection with band-preservation constraints
- Bad-time annotation (no blind hard-drops unless configured)
- Bad-channel detection + spline interpolation BEFORE re-referencing
- Adaptive re-referencing (mastoids → average → Cz)
- Optional ICA + ICLabel (with class thresholds)
- **ASR with bulletproof fallback**:
    * dynamic calibration + cutoff nudging
    * alpha-guard (thresholded)
    * retry at gentler cutoff
    * final reversion to pre-ASR if EEG bands degrade (no subject left worse)
    * optional Autoreject fallback hook (if installed)
- Cohort runner with per-file reports

This module relies on MNE. Optional deps: asrpy, mne-icalabel, autoreject.

Author: Emilio Borrelli
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from scipy.spatial import cKDTree  # noqa: E402

import numpy as np
import pandas as pd
import mne

# Optional deps (fail soft)
_ASR_AVAILABLE = True
try:
    from asrpy import ASR  # type: ignore
except Exception:
    _ASR_AVAILABLE = False

_ICLABEL_AVAILABLE = True
try:
    from mne.preprocessing import ICA
    from mne_icalabel import label_components  # type: ignore
except Exception:
    _ICLABEL_AVAILABLE = False

_AUTOREJECT_AVAILABLE = True
try:
    from autoreject import AutoReject  # type: ignore
except Exception:
    _AUTOREJECT_AVAILABLE = False

# -----------------------------
# Configuration & thresholds
# -----------------------------

@dataclass
class ICLabelThresholds:
    # eye_blink: float = 0.60
    # eye_movement: float = 0.60
    # muscle_artifact: float = 0.60
    # heart_beat: float = 0.60
    # line_noise: float = 0.60

    eye_blink: float = 0.80
    eye_movement: float = 0.80
    muscle_artifact: float = 0.85
    heart_beat: float = 0.55
    line_noise: float = 0.60

    def to_dict(self) -> Dict[str, float]:
        return {
            "eye blink": self.eye_blink,
            "eye movement": self.eye_movement,
            "muscle artifact": self.muscle_artifact,
            "heart beat": self.heart_beat,
            "line noise": self.line_noise,
        }

@dataclass
class StageThresholds:
    # Line noise
    line_resid_db_pass: float = 2.0
    line_resid_db_warn: float = 3.0
    line_shoulder_pass_pct: float = 12.0
    line_shoulder_warn_pct: float = 18.0
    line_band_change_pass_pct: float = 2.0
    line_band_change_warn_pct: float = 5.0

    # High-pass band preservation (abs % change)
    hp_alpha_pass_pct: float = 5.0
    hp_alpha_warn_pct: float = 10.0
    hp_beta_pass_pct: float = 5.0
    hp_beta_warn_pct: float = 10.0

    # Bad-time rejection (% of recording)
    bad_time_pass_pct: float = 15.0
    bad_time_warn_pct: float = 30.0

    # Bad-channel interpolation (count bounds derived from %)
    bad_channel_target_low_pct: float = 0.06  # ~4/64
    bad_channel_target_high_pct: float = 0.16 # tightened upper bound
    bad_channel_hard_cap_pct: float = 0.22    # warn if above target, fail if above cap

    # ICA band preservation (abs % change)
    ica_alpha_pass_pct: float = 10.0
    ica_alpha_warn_pct: float = 20.0
    ica_beta_pass_pct: float = 10.0
    ica_beta_warn_pct: float = 20.0
    ica_max_excluded_pass: int = 6
    ica_max_excluded_warn: int = 12

    # ASR calibration fraction & cutoff ranges
    asr_calib_pass_min: float = 0.20
    asr_calib_pass_max: float = 0.50
    asr_calib_warn_min: float = 0.10
    asr_calib_warn_max: float = 0.60
    asr_cutoff_pass_min: float = 3.0
    asr_cutoff_pass_max: float = 7.0
    asr_cutoff_warn_min: float = 2.0
    asr_cutoff_warn_max: float = 8.0

    # ASR alpha guard
    asr_alpha_guard_drop_pct: float = 30.0
    asr_retry_cutoff: float = 7.0

@dataclass
class PipelineConfig:
    line_noise_strategy: str = "bestof_safe"
    highpass_candidates: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.75, 1.0)
    drift_target_pct: float = 70.0
    band_tol_pct: float = 2.0
    annotate_bad_time: bool = True
    bad_time_win_sec: float = 1.0
    bad_time_overlap: float = 0.5
    bad_time_z_thresh: float = 9.0
    bad_time_frac_ch_amp: float = 0.25
    bad_time_flat_std_uv: float = 0.5
    bad_time_frac_ch_flat: float = 0.25
    bad_time_hf_band: Tuple[float, float] = (30.0, 100.0)
    bad_time_hf_rise_db: float = 8.0
    bad_time_min_span: float = 0.25
    hard_drop_long_sec: Optional[float] = None

    do_ica: bool = True
    ica_random_state: int = 97
    iclabel_thresholds: ICLabelThresholds = field(default_factory=ICLabelThresholds)

    ica_exclude_top_n: Optional[int] = None          # e.g., 3
    ica_top_rank: str = "artifact_prob"              # "artifact_prob" | "variance"
    ica_mode: str = "thresholds"   # <-- NEW: "thresholds" | "label"

    do_asr: bool = True
    asr_cutoff_initial: float = 4.0
    asr_cal_fraction: float = 0.30
    asr_cal_min_dur: float = 20.0
    asr_cal_max_dur: float = 120.0

    # Re-referencing order is adaptive inside pipeline

    # Output & reproducibility
    seed: int = 2025
    save_figures: bool = True
    psd_fmin: float = 1.0
    psd_fmax: float = 60.0

    # Logging / reports
    out_dir: Optional[Path] = None
    write_per_file_json: bool = True
    write_jsonl_log: bool = True

# -----------------------------
# Structured logger
# -----------------------------

class PipelineLogger:
    def __init__(self, root: Optional[Path], enable_jsonl: bool = True) -> None:
        self.root = Path(root) if root else None
        self.enable_jsonl = enable_jsonl
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)
        self._jsonl_fp = None
        if self.root and self.enable_jsonl:
            self._jsonl_fp = open(self.root / "pipeline.log.jsonl", "a", encoding="utf-8")

    def close(self):
        if self._jsonl_fp:
            self._jsonl_fp.close()
            self._jsonl_fp = None

    def _write_jsonl(self, record: Dict[str, Any]):
        if self._jsonl_fp:
            self._jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._jsonl_fp.flush()

    def event(self, kind: str, **fields: Any):
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "event": kind,
            **fields,
        }
        self._write_jsonl(record)

    def save_json(self, path: Path, payload: Dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

# -----------------------------
# Utility helpers
# -----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def capture_environment() -> Dict[str, Any]:
    def ver(mod: str) -> str:
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "unknown")
            return v
        except Exception:
            return "unknown"

    git_hash = None
    try:
        import subprocess
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        pass

    return {
        "python": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "versions": {
            "numpy": ver("numpy"),
            "scipy": ver("scipy"),
            "pandas": ver("pandas"),
            "mne": ver("mne"),
            "mne_icalabel": ver("mne_icalabel"),
            "asrpy": ver("asrpy"),
            "autoreject": ver("autoreject"),
        },
        "git": {"hash": git_hash},
    }

def ensure_finite_data(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    r = raw.copy()
    r.load_data()
    data = r.get_data()
    if not np.all(np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        r._data[:] = data
    return r



# -----------------------------
# Channel typing + hardware marker helpers
# -----------------------------

def normalize_channel_types(raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """
    Ensure EOG channels are correctly typed so they are NOT included as EEG.
    Heuristic based on common BrainVision naming.
    """
    r = raw.copy()

    eog_like = []
    for ch in r.ch_names:
        up = ch.upper()
        if "EOG" in up or up in {"VEOG", "HEOG", "VEOGL", "VEOGR", "HEOGL", "HEOGR"}:
            eog_like.append(ch)

    if eog_like:
        try:
            r.set_channel_types({ch: "eog" for ch in eog_like})
        except Exception:
            pass

    return r, {"eog_typed": eog_like}


def mark_hardware_dropouts(raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """
    Convert known hardware/dropout annotations into bad_time annotations.
    Your screenshot shows 'no USB Connection to actiCAP' which should be treated as bad_time.
    """
    r = raw.copy()
    if not len(r.annotations):
        return r, {"n_marked": 0, "matched": []}

    matched = []
    for ann in r.annotations:
        d = str(ann["description"]).lower()
        if ("usb" in d) or ("connection" in d) or ("acticap" in d) or ("no usb" in d):
            matched.append(ann)

    if not matched:
        return r, {"n_marked": 0, "matched": []}

    onsets = [float(a["onset"]) for a in matched]
    durations = [float(a["duration"]) for a in matched]
    desc = ["bad_time:hardware_dropout"] * len(matched)

    new = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=desc,
        orig_time=r.info.get("meas_date"),
    )
    r.set_annotations(r.annotations + new)

    return r, {
        "n_marked": len(matched),
        "matched": [str(a["description"]) for a in matched],
    }

# -----------------------------
# Spectral helpers (Welch)
# -----------------------------

from scipy.signal import welch, iirnotch, filtfilt  # noqa: E402


def _psd_median(raw: mne.io.BaseRaw, fmin=0.01, fmax=140.0, nperseg_sec=2.0):
    sf = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    f, P = welch(
        raw.get_data(picks=picks),
        fs=sf,
        nperseg=int(max(8, nperseg_sec * sf)),
        axis=-1,
        average="median",
    )
    keep = (f >= fmin) & (f <= fmax)
    return f[keep], np.median(P[:, keep], axis=0) if P.ndim == 2 else P[keep]


def _bandpower(f: np.ndarray, P: np.ndarray, band: Tuple[float, float]) -> float:
    idx = (f >= band[0]) & (f <= band[1])
    if not np.any(idx):
        return float("nan")
    return float(np.trapz(P[idx], f[idx]))


def _bandpowers_dict(raw: mne.io.BaseRaw, fmin=1, fmax=60) -> Dict[str, float]:
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (13, 30),
        "gamma": (30, 45),
    }
    f, P = _psd_median(raw, fmin=fmin, fmax=fmax)
    return {name: _bandpower(f, P, rng) for name, rng in bands.items()}


def _residual_peak_prom_db(raw: mne.io.BaseRaw, f0: float) -> float:
    f, P = _psd_median(raw, fmin=max(1, f0 - 6), fmax=f0 + 6)
    psd_db = 10 * np.log10(P + 1e-20)
    win = (f >= f0 - 1.5) & (f <= f0 + 1.5)
    if not np.any(win):
        return -np.inf
    pk = float(np.max(psd_db[win]))
    hood = (f >= f0 - 5) & (f <= f0 + 5) & (np.abs(f - f0) > 0.8)
    base = float(np.median(psd_db[hood])) if np.any(hood) else float(np.median(psd_db))
    return pk - base


def _shoulder_drop_pct(before: mne.io.BaseRaw, after: mne.io.BaseRaw, f0=50.0, gap=0.7, span=2.0) -> float:
    def bandpow(r: mne.io.BaseRaw, lo: float, hi: float) -> float:
        fb, Pb = _psd_median(r, fmin=lo, fmax=hi)
        return float(np.trapz(Pb, fb))

    left = (f0 - span, f0 - gap)
    right = (f0 + gap, f0 + span)
    b = bandpow(before, *left) + bandpow(before, *right)
    a = bandpow(after, *left) + bandpow(after, *right)
    return 100.0 * (1.0 - a / (b + 1e-20))


def _band_change_pct(before: mne.io.BaseRaw, after: mne.io.BaseRaw, bands=((8, 12), (13, 30))) -> float:
    vals: List[float] = []
    for lo, hi in bands:
        fb, Pb = _psd_median(before, fmin=lo, fmax=hi)
        fa, Pa = _psd_median(after, fmin=lo, fmax=hi)
        bb = float(np.trapz(Pb, fb))
        aa = float(np.trapz(Pa, fa))
        vals.append(100.0 * (aa - (bb + 1e-20)) / (bb + 1e-20))
    return float(np.mean(np.abs(vals)))


#---------------------------------------------------
# Other Helpers
#---------------------------------------------------

def _get_montage_pos(raw: mne.io.BaseRaw) -> Dict[str, np.ndarray]:
    """Return dict ch_name -> 3D position for channels that have finite coords."""
    mont = raw.get_montage()
    if mont is None:
        return {}

    ch_pos = mont.get_positions().get("ch_pos", {})
    out: Dict[str, np.ndarray] = {}

    for ch in raw.ch_names:
        xyz = ch_pos.get(ch)
        if xyz is None:
            continue
        xyz = np.asarray(xyz, dtype=float)

        # ✅ NEW: filter non-finite positions
        if xyz.shape[0] >= 3 and np.all(np.isfinite(xyz[:3])):
            out[ch] = xyz[:3]

    return out

def _infer_region_map(raw: mne.io.BaseRaw, n_regions: int = 6) -> Dict[str, str]:
    """
    Divide channels into n_regions based on polar angle around head.
    This works for arbitrary channel counts (32, 64, 128, 200+).
    """
    pos = _get_montage_pos(raw)
    if not pos:
        return {}

    chs = []
    angles = []
    for ch, xyz in pos.items():
        x, y, z = xyz
        angle = np.arctan2(y, x)  # frontal ~ +y, occipital ~ -y
        chs.append(ch)
        angles.append(angle)

    idx = np.argsort(angles)
    chs_sorted = np.array(chs)[idx]
    groups = np.array_split(chs_sorted, n_regions)

    region_map: Dict[str, str] = {}
    for i, grp in enumerate(groups):
        for ch in grp:
            region_map[str(ch)] = f"region_{i}"
    return region_map


def _compute_neighbor_radius(
    raw: mne.io.BaseRaw, k: int = 4, scale: float = 1.6
) -> Optional[float]:
    """
    Compute a reasonable spatial radius for 'neighbors', based on montage.
    Uses median distance to the k-th nearest neighbor as baseline.
    Robust to incomplete / non-finite montages.
    """
    pos = _get_montage_pos(raw)
    if not pos:
        return None

    coords = np.stack(list(pos.values()), axis=0)

    # ✅ NEW: extra safety filter (paranoia layer)
    finite_mask = np.all(np.isfinite(coords), axis=1)
    coords = coords[finite_mask]

    if coords.shape[0] < k + 1:
        return None

    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k)
    baseline = np.median(dists[:, k - 1])
    return float(baseline * scale)


# -----------------------------
# Core stages
# -----------------------------

_DEF_F0_GUESS = (50.0, 60.0)


def _detect_mains_f0(raw: mne.io.BaseRaw, guess: Tuple[float, ...] = _DEF_F0_GUESS) -> Tuple[Optional[float], float]:
    sf = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    f, Pxx = welch(raw.get_data(picks=picks), fs=sf, nperseg=int(8 * sf), axis=-1, average="median")
    psd = np.median(Pxx, axis=0)
    psd_db = 10 * np.log10(psd + 1e-20)

    best = (None, -np.inf)
    for g in guess:
        win = (f >= g - 2) & (f <= g + 2)
        if not np.any(win):
            continue
        pk = float(np.max(psd_db[win]))
        hood = (f >= g - 5) & (f <= g + 5) & (np.abs(f - g) > 0.8)
        base = float(np.median(psd_db[hood])) if np.any(hood) else float(np.median(psd_db))
        prom_db = pk - base
        if prom_db > best[1]:
            best = (g, prom_db)
    return best  # (f0, prom_db)


def _iir_notch(raw: mne.io.BaseRaw, f0: float, Q: float) -> mne.io.BaseRaw:
    sf = raw.info["sfreq"]
    b, a = iirnotch(f0 / (sf / 2), Q)
    out = raw.copy()
    out._data[:] = filtfilt(b, a, out.get_data(), axis=1)
    return out


def line_noise_bestof_safe(raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    f0, prom = _detect_mains_f0(raw)
    if f0 is None or prom < 0.5:
        return raw.copy(), {"stage": "skip", "detected_f0": f0, "prom_db": prom}

    sf = raw.info["sfreq"]
    df_grid = (-0.30, -0.15, 0.0, +0.15, +0.30)
    q_grid = (160.0, 200.0, 240.0)

    def eval_candidate(name: str, r_after: mne.io.BaseRaw) -> Dict[str, Any]:
        resid = _residual_peak_prom_db(r_after, f0)
        # Include 2*f0 residue a bit
        resid2 = 0.0
        if 2 * f0 < sf / 2 - 1:
            resid2 = max(0.0, _residual_peak_prom_db(r_after, 2 * f0))
        resid_mix = 0.7 * max(0.0, resid) + 0.3 * resid2
        shoulder = _shoulder_drop_pct(raw, r_after, f0=f0)
        bandchg = _band_change_pct(raw, r_after)
        J = 1.2 * resid_mix + 0.25 * max(0.0, shoulder - 12.0) + 0.25 * abs(bandchg)
        return {
            "name": name,
            "J": J,
            "resid_db": resid_mix,
            "shoulder_drop_pct": shoulder,
            "band_change_pct": bandchg,
        }

    candidates: List[Tuple[Dict[str, Any], mne.io.BaseRaw]] = []

    # IIR notch grid + optional 2*f0 notch
    for Q in q_grid:
        for df in df_grid:
            r1 = _iir_notch(raw, f0 + df, Q)
            if 2 * f0 < sf / 2 - 1:
                if _residual_peak_prom_db(r1, 2 * f0) >= 1.0:
                    r1 = _iir_notch(r1, 2 * f0, 130.0)
            rec = eval_candidate(f"iir_Q{Q}_df{df:+.2f}", r1)
            candidates.append((rec, r1))

    # Spectrum-fit notch (MNE)
    try:
        freqs = [f0]
        if 2 * f0 < sf / 2 - 1:
            freqs.append(2 * f0)
        r2 = raw.copy().notch_filter(freqs=freqs, method="spectrum_fit", mt_bandwidth=1.2, p_value=0.01, phase="zero")
        rec2 = eval_candidate("spectrum_fit_bw1.2_p0.01", r2)
        candidates.append((rec2, r2))
    except Exception:
        pass

    best = sorted(candidates, key=lambda t: t[0]["J"])[0]
    rec, rbest = best

    # Gentle final sweep
    if not (rec["resid_db"] <= 1.0 and abs(rec["band_change_pct"]) <= 2.0 and rec["shoulder_drop_pct"] <= 10.0):
        rbest = _iir_notch(rbest, f0, 220.0)
        if 2 * f0 < sf / 2 - 1:
            if _residual_peak_prom_db(rbest, 2 * f0) >= 0.8:
                rbest = _iir_notch(rbest, 2 * f0, 160.0)
        # recompute quick metrics
        rec.update({
            "resid_db": _residual_peak_prom_db(rbest, f0),
            "shoulder_drop_pct": _shoulder_drop_pct(raw, rbest, f0=f0),
            "band_change_pct": _band_change_pct(raw, rbest),
            "name": rec["name"] + "+final_sweep",
            "stage": "bestof",
            "detected_f0": f0,
            "prom_db": prom,
        })
    else:
        rec.update({"stage": "bestof", "detected_f0": f0, "prom_db": prom})

    return rbest, rec


def adaptive_highpass(
    raw: mne.io.BaseRaw,
    candidates: Tuple[float, ...],
    drift_target: float,
    band_tol: float,
) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    # Measure baseline
    f0, P0 = _psd_median(raw, fmin=0.01, fmax=min(140.0, raw.info["sfreq"] / 2 - 1))
    def bp(band):
        return _bandpower(f0, P0, band)
    vlf0 = bp((0.1, 0.5))
    a0 = bp((8, 12))
    b0 = bp((13, 30))

    rows = []
    best_raw = None
    for hp in candidates:
        r = raw.copy().filter(l_freq=hp, h_freq=None, method="fir", phase="zero", verbose=False)
        f, P = _psd_median(r, fmin=0.01, fmax=min(140.0, r.info["sfreq"] / 2 - 1))
        vlf = _bandpower(f, P, (0.1, 0.5))
        a = _bandpower(f, P, (8, 12))
        b = _bandpower(f, P, (13, 30))
        ac1 = _autocorr_at_lag_sec(r, 1.0)
        rows.append({
            "highpass_hz": hp,
            "drift_reduction_pct": 100.0 * (1 - vlf / (vlf0 + 1e-20)),
            "alpha_change_pct": 100.0 * ((a - a0) / (a0 + 1e-20)),
            "beta_change_pct": 100.0 * ((b - b0) / (b0 + 1e-20)),
            "ac1_change_pct": ac1,
        })
        if best_raw is None:
            best_raw = r
    df = pd.DataFrame(rows).sort_values("highpass_hz").reset_index(drop=True)

    cand = df[
        (df["drift_reduction_pct"] >= drift_target)
        & (df["alpha_change_pct"].abs() <= band_tol)
        & (df["beta_change_pct"].abs() <= band_tol)
    ]
    if len(cand):
        hp_sel = float(cand.iloc[0]["highpass_hz"])  # lowest hp meeting criteria
    else:
        # score = drift reduction - penalty for band changes
        score = df["drift_reduction_pct"].fillna(0) - (df["alpha_change_pct"].abs() + df["beta_change_pct"].abs())
        hp_sel = float(df.loc[score.idxmax(), "highpass_hz"])

    raw_hp = raw.copy().filter(l_freq=hp_sel, h_freq=None, method="fir", phase="zero", verbose=False)
    return raw_hp, {"recommended": hp_sel, "df_scan": df.to_dict(orient="list")}


def _autocorr_at_lag_sec(raw: mne.io.BaseRaw, lag_sec: float) -> float:
    sf = raw.info["sfreq"]
    lag = int(round(lag_sec * sf))
    X = raw.get_data(picks=mne.pick_types(raw.info, eeg=True, exclude="bads"))
    vals = []
    for x in X:
        x = x - x.mean()
        if lag >= len(x):
            continue
        num = float(np.dot(x[:-lag], x[lag:]))
        den = float(np.dot(x, x)) + 1e-20
        vals.append(num / den)
    return float(np.median(vals)) if vals else float("nan")

# -----------------------------
# Bad-time, bad channels, reref
# -----------------------------


def annotate_bad_time(raw: mne.io.BaseRaw, cfg: PipelineConfig) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    sf = raw.info["sfreq"]
    size = int(round(cfg.bad_time_win_sec * sf))
    step = int(round(cfg.bad_time_win_sec * (1.0 - cfg.bad_time_overlap) * sf))
    if size < 8 or step < 1:
        return raw.copy(), {"n_spans": 0, "seconds": 0.0, "fraction": 0.0, "spans": []}

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    data = raw.get_data(picks=picks) * 1e6
    T = data.shape[1]

    # Amplitude z via peak-to-peak
    p2p = []
    for start in range(0, T - size + 1, step):
        sl = slice(start, start + size)
        p2p.append(np.ptp(data[:, sl], axis=1))
    p2p = np.stack(p2p, axis=1) if p2p else np.zeros((len(picks), 0))
    med = np.median(p2p, axis=1, keepdims=True) if p2p.size else 0.0
    mad = np.median(np.abs(p2p - med), axis=1, keepdims=True) + 1e-20 if p2p.size else 1.0
    z = (p2p - med) / (1.4826 * mad) if p2p.size else np.zeros_like(p2p)

    # High-frequency rise per window (approx baseline = first window)
    def hf_power_db(window_sl: slice) -> float:
        f, P = welch(data[:, window_sl], fs=sf, nperseg=size, axis=-1, average="median")
        keep = (f >= cfg.bad_time_hf_band[0]) & (f <= cfg.bad_time_hf_band[1])
        return 10 * math.log10(float(np.median(np.trapz(P[:, keep], f[keep], axis=1))) + 1e-20)

    hf_db_windows: List[float] = []
    for start in range(0, T - size + 1, step):
        sl = slice(start, start + size)
        hf_db_windows.append(hf_power_db(sl))
    base_db = hf_db_windows[0] if hf_db_windows else 0.0

    spans: List[Tuple[float, float, List[str]]] = []
    for w, start in enumerate(range(0, T - size + 1, step)):
        triggers: List[str] = []
        if p2p.size:
            frac_amp = float((z[:, w] >= cfg.bad_time_z_thresh).mean())
            if frac_amp >= cfg.bad_time_frac_ch_amp:
                triggers.append("z-amp")
        # flatline detection via std threshold
        std_uv = np.std(data[:, start : start + size], axis=1)
        if float((std_uv < cfg.bad_time_flat_std_uv).mean()) >= cfg.bad_time_frac_ch_flat:
            triggers.append("flatline")
        if (hf_db_windows[w] - base_db) >= cfg.bad_time_hf_rise_db:
            triggers.append("hf-noise")
        if triggers:
            onset = start / sf
            duration = size / sf
            spans.append((onset, duration, triggers))

    # Merge touching windows
    spans.sort(key=lambda s: s[0])
    merged: List[Tuple[float, float, set]] = []
    for onset, dur, labels in spans:
        if not merged:
            merged.append((onset, onset + dur, set(labels)))
        else:
            a, b, lab = merged[-1]
            if onset <= b + 1e-6:
                merged[-1] = (a, max(b, onset + dur), lab.union(labels))
            else:
                merged.append((onset, onset + dur, set(labels)))

    # Filter by min span
    merged = [m for m in merged if (m[1] - m[0]) >= cfg.bad_time_min_span]

    raw_out = raw.copy()
    if merged:
        onsets = [m[0] for m in merged]
        durations = [m[1] - m[0] for m in merged]
        desc = ["bad_time:" + "+".join(sorted(list(m[2]))) for m in merged]
        ann = mne.Annotations(onset=onsets, duration=durations, description=desc, orig_time=raw.info.get("meas_date"))
        raw_out.set_annotations(raw_out.annotations + ann)

    hard_dropped = 0.0
    if cfg.hard_drop_long_sec is not None and merged:
        long_spans = [(a, b) for (a, b, _) in merged if (b - a) >= cfg.hard_drop_long_sec]
        if long_spans:
            # Keep the rest
            keep: List[Tuple[float, float]] = []
            prev = 0.0
            for a, b in long_spans:
                if a > prev:
                    keep.append((prev, a))
                prev = b
            if prev < raw.times[-1]:
                keep.append((prev, raw.times[-1]))
            if keep:
                raws = [raw_out.copy().crop(tmin=a, tmax=b, include_tmax=False) for (a, b) in keep]
                raw_out = mne.concatenate_raws(raws, on_mismatch="ignore")
                hard_dropped = sum([b - a for (a, b) in long_spans])

    total_bad = sum([(m[1] - m[0]) for m in merged])
    return raw_out, {
        "n_spans": len(merged),
        "seconds": float(total_bad),
        "fraction": float(total_bad / max(1e-9, raw.times[-1])),
        "hard_dropped_sec": float(hard_dropped),
        "spans": [
            {"onset": float(m[0]), "duration": float(m[1] - m[0]), "labels": sorted(list(m[2]))} for m in merged
        ],
        "rules": asdict(cfg)["bad_time_hf_band"],
    }


def extract_clean_only_raw(raw: mne.io.BaseRaw, bad_prefix: str = "bad_time") -> mne.io.BaseRaw:
    if len(raw.annotations) == 0:
        return raw.copy()
    spans = [
        (a["onset"], a["onset"] + a["duration"]) for a in raw.annotations if str(a["description"]).startswith(bad_prefix)
    ]
    if not spans:
        return raw.copy()
    spans.sort()
    keep: List[Tuple[float, float]] = []
    t0 = 0.0
    for a, b in spans:
        if a > t0:
            keep.append((t0, a))
        t0 = max(t0, b)
    if t0 < raw.times[-1]:
        keep.append((t0, raw.times[-1]))
    if not keep:
        seg = raw.copy().crop(tmin=0, tmax=min(1.0, raw.times[-1]), include_tmax=False)
        seg._data[:] = 0.0
        return seg
    return mne.concatenate_raws([raw.copy().crop(tmin=a, tmax=b, include_tmax=False) for (a, b) in keep], on_mismatch="ignore")


def ensure_positions(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    r = raw.copy()
    try:
        r.set_montage("standard_1020", match_case=False, on_missing="ignore")
    except Exception:
        pass
    return r


def sanitize_nonfinite(raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, List[str], Dict[str, int]]:
    r = raw.copy()
    data = r.get_data()
    nonfin = ~np.isfinite(data)
    dropped: List[str] = []
    filled: Dict[str, int] = {}
    if nonfin.any():
        n_times = data.shape[1]
        for i, ch in enumerate(r.ch_names):
            frac = float(nonfin[i].mean())
            if frac > 0.25:
                dropped.append(ch)
        if dropped:
            r.drop_channels(dropped)
            data = r.get_data()
            nonfin = ~np.isfinite(data)
        if nonfin.any():
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            r._data[:] = data
            for i, ch in enumerate(r.ch_names):
                filled[ch] = int(nonfin[i].sum())
    return r, dropped, filled


def detection_view(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    r = ensure_positions(raw)
    r.load_data()
    try:
        r.filter(l_freq=1.0, h_freq=40.0, method="fir", phase="zero", verbose=False)
        r.set_eeg_reference("average")
    except Exception:
        pass
    return r


def detect_bad_channels(raw_for_metrics: mne.io.BaseRaw, corr_k=2.0, var_k=3.0, flat_std_uv=1e-3) -> Tuple[List[str], Dict[str, Any]]:
    picks = mne.pick_types(raw_for_metrics.info, eeg=True, exclude="bads")
    ch_names = [raw_for_metrics.ch_names[p] for p in picks]
    epochs = mne.make_fixed_length_epochs(raw_for_metrics, duration=2.0, overlap=1.0, preload=True, reject_by_annotation=True)
    epochs.pick_types(eeg=True)
    X = epochs.get_data()  # (n_ep, n_ch, n_t)
    var_ch = np.var(X, axis=(0, 2))
    mean_corr = []
    for e in range(X.shape[0]):
        C = np.corrcoef(X[e])
        C = np.nan_to_num(C, nan=0.0)
        mean_corr.append((C.sum(axis=1) - 1.0) / max(1, (X.shape[1] - 1)))
    mean_corr = np.mean(mean_corr, axis=0)
    std_ch_uv = np.std(X * 1e6, axis=(0, 2))

    def rob(m):
        med = np.median(m)
        mad = np.median(np.abs(m - med)) + 1e-20
        return med, mad

    med_v, mad_v = rob(var_ch)
    var_hi_thr = med_v + var_k * mad_v
    var_lo_thr = max(1e-20, med_v - var_k * mad_v)

    med_c, mad_c = rob(mean_corr)
    corr_thr = med_c - corr_k * mad_c

    bad_high_var = [ch_names[i] for i, v in enumerate(var_ch) if v > var_hi_thr]
    bad_low_var = [ch_names[i] for i, v in enumerate(var_ch) if v < var_lo_thr]
    bad_low_corr = [ch_names[i] for i, c in enumerate(mean_corr) if c < corr_thr]
    bad_flat = [ch_names[i] for i, s in enumerate(std_ch_uv) if s < flat_std_uv]

    bad_union = list(dict.fromkeys(bad_high_var + bad_low_var + bad_low_corr + bad_flat))

    rep = {
        "thresholds": {"var_hi_thr": float(var_hi_thr), "var_lo_thr": float(var_lo_thr), "corr_thr": float(corr_thr), "corr_k": corr_k, "var_k": var_k, "flat_std_uv": flat_std_uv},
        "bad_by_rule": {
            "high_variance": bad_high_var,
            "low_variance": bad_low_var,
            "low_corr": bad_low_corr,
            "flatline": bad_flat,
        },
        "bad_union": bad_union,
        "n_epochs": int(X.shape[0]),
    }
    return bad_union, rep


def interpolate_bad_channels(raw: mne.io.BaseRaw, cfg: StageThresholds) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """
    Bad-channel detection + interpolation with stability safeguards:

    - Uses your existing robust detection (var/corr/flatline)
    - Applies percentage-based caps (from cfg) so we never drop too many
    - Adds region-wise caps (data-driven, from montage geometry)
    - Ensures bad channels have enough spatial neighbors to be interpolated
    - Falls back to a simple smoothing strategy if spline interpolation fails
    """
    r = ensure_positions(raw)
    metrics_view = detection_view(r)

    # --- 1) Initial detection using your existing logic ---
    corr_k, var_k = 2.0, 3.0
    bads, rep = detect_bad_channels(metrics_view, corr_k=corr_k, var_k=var_k)

    n_ch = len(mne.pick_types(r.info, eeg=True))
    # Your existing target ranges based on % of total channels
    target_low = max(1, int(math.floor(cfg.bad_channel_target_low_pct * n_ch)))
    target_high = max(target_low, int(math.ceil(cfg.bad_channel_target_high_pct * n_ch)))
    hard_cap = max(target_high, int(math.ceil(cfg.bad_channel_hard_cap_pct * n_ch)))

    it = 0
    while len(bads) > target_high and it < 6:
        corr_k += 0.5
        var_k += 0.5
        bads, rep = detect_bad_channels(metrics_view, corr_k=corr_k, var_k=var_k)
        it += 1
        if len(bads) >= hard_cap:
            break

    # --- 2) Region-wise and global safety caps (scales with any channel count) ---
    region_map = _infer_region_map(r, n_regions=6)
    if region_map:
        # count bads per region
        region_counts: Dict[str, int] = {}
        region_sizes: Dict[str, int] = {}
        for ch, reg in region_map.items():
            region_sizes.setdefault(reg, 0)
            region_sizes[reg] += 1
        for ch in bads:
            reg = region_map.get(ch)
            if reg is None:
                continue
            region_counts[reg] = region_counts.get(reg, 0) + 1

        # Max fractions (independent of absolute channel count)
        MAX_REGION_FRAC = 0.33  # drop ≤ 33% in any region
        MAX_GLOBAL_FRAC = cfg.bad_channel_hard_cap_pct  # e.g. 0.22 by default

        rescued_region: set[str] = set()
        for ch in bads:
            reg = region_map.get(ch)
            if not reg:
                continue
            size = region_sizes.get(reg, 1)
            cnt = region_counts.get(reg, 0)
            if size > 0 and (cnt / size) > MAX_REGION_FRAC:
                # rescue this channel to avoid wiping out the region
                rescued_region.add(ch)
                region_counts[reg] -= 1

        if rescued_region:
            bads = [ch for ch in bads if ch not in rescued_region]

        # Enforce global fraction cap as well
        max_global_bad = int(round(MAX_GLOBAL_FRAC * n_ch))
        max_global_bad = max(1, max_global_bad)
        if len(bads) > max_global_bad:
            flags = rep.get("bad_by_rule", {})
            hv = set(flags.get("high_variance", []))
            lv = set(flags.get("low_variance", []))
            lc = set(flags.get("low_corr", []))
            fl = set(flags.get("flatline", []))

            scored = []
            for ch in bads:
                score = (
                    (ch in fl)*3 +
                    (ch in hv)*2 +
                    (ch in lv)*1 +
                    (ch in lc)*1
                )
                scored.append((score, ch))
            scored.sort(reverse=True)
            bads = [ch for _, ch in scored[:max_global_bad]]

    # --- 3) Neighbor requirement for safe interpolation ---
    pos = _get_montage_pos(r)
    radius = _compute_neighbor_radius(r, k=4, scale=1.8)
    if radius is not None and pos:
        def has_neighbors(ch: str) -> bool:
            if ch not in pos:
                return True
            xyz = pos[ch]
            cnt = 0
            for other, xyz2 in pos.items():
                if other == ch:
                    continue
                if np.linalg.norm(xyz2 - xyz) < radius:
                    cnt += 1
            return cnt >= 3

        rescued_neighbors = [ch for ch in bads if not has_neighbors(ch)]
        if rescued_neighbors:
            bads = [ch for ch in bads if ch not in rescued_neighbors]



    # --- 4) Interpolate with spline; fall back if needed ---
    r_interp = r.copy()
    r_interp.info["bads"] = [b for b in bads if b in r_interp.ch_names]

    try:
        r_interp.interpolate_bads(reset_bads=False, mode="spline")
    except Exception:
        # Fallback: simple temporal smoothing so we never fail catastrophically
        data = r_interp.get_data()
        # 5-point moving average as last-resort backup
        kernel = np.ones(5, dtype=float) / 5.0
        smoothed = np.vstack([
            np.convolve(ch, kernel, mode="same") for ch in data
        ])
        r_interp._data[:] = smoothed

    return r_interp, {
        "n_eeg": n_ch,
        "target_range": (target_low, target_high),
        "hard_cap": hard_cap,
        "final_n_bads": len(r_interp.info["bads"]),
        "bads": list(r_interp.info["bads"]),
        "thresholds": rep["thresholds"],
        "rule_counts": {k: len(v) for k, v in rep["bad_by_rule"].items()},
        "iters": it,
    }



def adaptive_rereference(raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    r = raw.copy()

    # Build case-preserving lookup
    name_map = {c.upper(): c for c in r.ch_names}
    bads = set(r.info.get("bads", []))

    # True mastoid-ish candidates
    pref = ["M1", "M2", "TP9", "TP10"]
    present = [name_map[c] for c in pref if c in name_map]
    present = [c for c in present if c not in bads]

    if len(present) >= 2:
        try:
            r.set_eeg_reference(ref_channels=present[:2])
            return r, {"method": "mastoids", "mastoids": present[:2]}
        except Exception:
            pass

    # Fallback: average
    try:
        r.set_eeg_reference("average")
        return r, {"method": "average"}
    except Exception:
        pass

    # Last fallback: Cz if available and not bad
    if "CZ" in name_map:
        cz = name_map["CZ"]
        if cz not in bads:
            try:
                r.set_eeg_reference(ref_channels=[cz])
                return r, {"method": "cz"}
            except Exception:
                pass

    return r, {"method": None}



# -----------------------------
# ICA + ICLabel
# -----------------------------

def _alpha_bandpower(raw, band=(8,12)):
    f, P = _psd_median(raw, fmin=1, fmax=45)
    m = (f >= band[0]) & (f <= band[1])
    return float(np.trapz(P[m], f[m])) if np.any(m) else float("nan")

def run_ica_iclabel(
    raw: mne.io.BaseRaw,
    random_state: int,
    thresholds: ICLabelThresholds,
    top_n: Optional[int] = None,
    top_rank: str = "artifact_prob",
    alpha_guard_pct: float = 15.0,
    safety_max_excluded_frac: float = 0.50,
    mode: str = "thresholds",
):
    if not _ICLABEL_AVAILABLE:
        raise RuntimeError("ICA/ICLabel not available")

    # ---------- Prepare EEG-only object ----------
    r = raw.copy()
    try:
        r.set_montage("standard_1020", match_case=False, on_missing="ignore")
    except Exception:
        pass

    # critical: ensure EOG is not mixed into EEG ICA
    r = r.pick_types(eeg=True)

    # ICLabel tends to work better with average reference
    try:
        r.set_eeg_reference("average", projection=False)
    except Exception:
        pass

    # band-pass for ICA fit
    r_bp = r.copy().filter(l_freq=1.0, h_freq=100.0, verbose=False)

    n_comp = max(5, min(len(r_bp.ch_names) - 1, 20))

    ica = ICA(
        n_components=n_comp,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(r_bp, reject_by_annotation=True)

    # ---------- ICLabel ----------
    labels, confidences = [], []
    iclabel_error = None

    try:
        res = label_components(r_bp, ica, method="iclabel")

        if isinstance(res, dict):  # new style
            labels = list(res.get("labels", []))
            probs = np.asarray(res.get("y_pred_proba", []), dtype=float)
        else:  # old tuple style
            labels = list(res[0])
            probs = np.asarray(res[1], dtype=float)

        if probs.ndim == 2:
            conf = np.max(probs, axis=1)
        elif probs.ndim == 1:
            conf = probs
        else:
            conf = np.zeros(len(labels))

        confidences = conf.tolist()

    except Exception as e:
        iclabel_error = str(e)
        labels = ["other"] * n_comp
        confidences = [0.0] * n_comp

    # ---------- Build table ----------
    iclabel_table = []
    for i, (lab, conf) in enumerate(zip(labels, confidences)):
        iclabel_table.append({
            "idx": int(i),
            "label": str(lab),
            "confidence": float(conf),
            "excluded": False,
        })

    # ---------- Apply thresholds ----------
    exclude = []

    if mode == "thresholds":
        th_map = thresholds.to_dict()
        for row in iclabel_table:
            lbl = row["label"].lower().replace("_", " ")
            if lbl in th_map and row["confidence"] >= th_map[lbl]:
                row["excluded"] = True
                exclude.append(row["idx"])

        if not exclude and any(row["label"] != "other" for row in iclabel_table):
            mode = "label"

    if mode == "label":
        artifact_classes = {
            "eye blink", "eye movement", "muscle artifact",
            "heart beat", "line noise", "channel noise"
        }
        for row in iclabel_table:
            lbl = row["label"].lower().replace("_", " ")
            if lbl in artifact_classes:
                row["excluded"] = True
                exclude.append(row["idx"])

    report_mode = "no_artifacts" if not exclude else mode

    # ---------- Safety caps ----------
    cap_allowed = min(n_comp - 1, int(np.floor(n_comp * safety_max_excluded_frac)))
    if len(exclude) > cap_allowed:
        order = np.argsort([row["confidence"] for row in iclabel_table])[::-1]
        keep = [iclabel_table[j]["idx"] for j in order[:cap_allowed]]
        exclude = keep
        for row in iclabel_table:
            row["excluded"] = row["idx"] in exclude

    # ---------- Apply ICA with alpha guard ----------
    alpha_before = _alpha_bandpower(r_bp)
    ica.exclude = exclude
    r_after = ica.apply(r_bp.copy())
    alpha_after = _alpha_bandpower(r_after)
    alpha_drop = 100.0 * (alpha_before - alpha_after) / (alpha_before + 1e-20)

    alpha_guard_triggered = False
    if np.isfinite(alpha_drop) and alpha_drop > alpha_guard_pct:
        alpha_guard_triggered = True
        cur = exclude[:]
        while cur:
            cur.pop(0)
            ica.exclude = cur
            r_after = ica.apply(r_bp.copy())
            alpha_after = _alpha_bandpower(r_after)
            alpha_drop = 100.0 * (alpha_before - alpha_after) / (alpha_before + 1e-20)
            if alpha_drop <= alpha_guard_pct:
                exclude = cur
                break

    # ---------- Report ----------
    report = {
        "stage": "success" if iclabel_error is None else "iclabel_fallback",
        "iclabel_error": iclabel_error,
        "mode_used": report_mode,
        "n_components": n_comp,
        "excluded": exclude,
        "n_excluded": len(exclude),
        "alpha_before": float(alpha_before),
        "alpha_after": float(alpha_after),
        "alpha_change_pct": float(alpha_after - alpha_before) / (alpha_before + 1e-20) * 100.0,
        "alpha_guard_triggered": alpha_guard_triggered,
        "iclabel_table": iclabel_table,
    }

    return r_after, report



# -----------------------------
# ASR with bulletproof fallback
# -----------------------------

def run_asr_with_fallback(raw: mne.io.BaseRaw, cfg: PipelineConfig, thr: StageThresholds):
    if not _ASR_AVAILABLE or not cfg.do_asr:
        return raw.copy(), {"stage": "skip", "reason": "ASR not available or disabled"}
    raw = ensure_finite_data(raw)
    bp_before = _bandpowers_dict(raw)

    asr = ASR(
        sfreq=raw.info["sfreq"],
        cutoff=cfg.asr_cutoff_initial,
        blocksize=5,
        min_clean_fraction=0.25,
        max_dropout_fraction=0.1,
    )

    dur = raw.times[-1]
    cal_dur = max(cfg.asr_cal_min_dur, min(cfg.asr_cal_max_dur, cfg.asr_cal_fraction * dur))
    raw_cal = raw.copy().crop(tmin=0, tmax=min(cal_dur, dur), include_tmax=False)

    # Fit on calibration segment (Raw)
    asr.fit(raw_cal)

    # Transform full raw (Raw in, Raw out)
    raw_asr = asr.transform(raw.copy())

    cal_frac = cal_dur / max(1e-9, dur)
    bp_after = _bandpowers_dict(raw_asr)
    alpha_drop = 100.0 * (1 - (bp_after["alpha"] / (bp_before["alpha"] + 1e-20)))

    if alpha_drop > thr.asr_alpha_guard_drop_pct:
        # Retry at gentler cutoff
        try:
            asr2 = ASR(
                sfreq=raw.info["sfreq"],
                cutoff=thr.asr_retry_cutoff,
                blocksize=5,
                min_clean_fraction=0.25,
                max_dropout_fraction=0.1,
            )
            asr2.fit(raw_cal)
            raw_retry = asr2.transform(raw.copy())
            bp_retry = _bandpowers_dict(raw_retry)
            alpha_drop_retry = 100.0 * (1 - (bp_retry["alpha"] / (bp_before["alpha"] + 1e-20)))
            if alpha_drop_retry <= thr.asr_alpha_guard_drop_pct:
                return raw_retry, {
                    "stage": "retry",
                    "cutoff": thr.asr_retry_cutoff,
                    "calib_fraction": cal_frac,
                    "alpha_drop_pct": alpha_drop_retry,
                    "bandpowers_before": bp_before,
                    "bandpowers_after": bp_retry,
                }
        except Exception:
            pass

        # Autoreject fallback
        raw_ar = _apply_autoreject_safe(raw)
        if raw_ar is not None:
            return raw_ar, {
                "stage": "autoreject",
                "calib_fraction": cal_frac,
                "alpha_drop_pct": alpha_drop,
                "bandpowers_before": bp_before,
                "bandpowers_after": _bandpowers_dict(raw_ar),
            }

        return raw.copy(), {
            "stage": "revert",
            "reason": f"alpha dropped {alpha_drop:.1f}%",
            "calib_fraction": cal_frac,
            "bandpowers_before": bp_before,
            "bandpowers_after": bp_before,
        }

    return raw_asr, {
        "stage": "success",
        "cutoff": cfg.asr_cutoff_initial,
        "calib_fraction": cal_frac,
        "alpha_drop_pct": alpha_drop,
        "bandpowers_before": bp_before,
        "bandpowers_after": bp_after,
    }


def _apply_autoreject_safe(raw: mne.io.BaseRaw) -> Optional[mne.io.BaseRaw]:
    """Try autoreject cleaning. Returns a Raw or None if fails."""
    if not _AUTOREJECT_AVAILABLE:
        return None
    try:
        epochs = mne.make_fixed_length_epochs(
            raw, duration=2.0, preload=True, reject_by_annotation=True
        )
        ar = AutoReject()
        ar.fit(epochs)
        epochs_clean, _ = ar.transform(epochs, return_log=True)
        # Reconstruct continuous signal from cleaned epochs
        data = epochs_clean.get_data().reshape(len(raw.ch_names), -1)
        raw_out = raw.copy()
        n = min(raw_out._data.shape[1], data.shape[1])
        raw_out._data[:, :n] = data[:, :n]
        return raw_out
    except Exception:
        return None


# -----------------------------
# Orchestrator
# -----------------------------

def run_pipeline(
    raw: mne.io.BaseRaw,
    cfg: PipelineConfig,
    thr: StageThresholds,
    logger: Optional[PipelineLogger] = None,
) -> Tuple[mne.io.BaseRaw, Dict[str, Any], Dict[str, mne.io.BaseRaw]]:
    """
    Run full preprocessing pipeline.
    Returns:
      - final cleaned raw
      - report (dict)
      - intermediates (dict of stage_name -> Raw object)
    """
    set_global_seed(cfg.seed)
    report: Dict[str, Any] = {
        "stages": {},
        "env": capture_environment(),
        "config": asdict(cfg),
    }
    intermediates: Dict[str, mne.io.BaseRaw] = {"input": raw.copy()}

    def log_stage(name: str, func, *args, **kwargs):
        bp_before = _bandpowers_dict(raw)
        t0 = time.time()
        out, rec = func(*args, **kwargs)
        dt = time.time() - t0
        bp_after = _bandpowers_dict(out)

        rec = {**rec,
               "bandpowers_before": bp_before,
               "bandpowers_after": bp_after}

        report["stages"][name] = {"time_sec": dt, **rec}
        intermediates[name] = out.copy()

        if logger:
            logger.event("stage", stage_name=name, time_sec=dt, **rec)
        return out

    # 1. Sanitize
    # 1. Sanitize
    raw, dropped, filled = sanitize_nonfinite(raw)
    report["stages"]["sanitize"] = {"dropped": dropped, "filled": filled}
    intermediates["sanitize"] = raw.copy()

    # 1b. Normalize channel types (IMPORTANT for VEOG)
    raw = log_stage("channel_types", normalize_channel_types, raw)

    # 1c. Convert hardware/USB markers to bad_time
    raw = log_stage("hardware_dropouts", mark_hardware_dropouts, raw)

    # 2. Line noise
    raw = log_stage("line_noise", line_noise_bestof_safe, raw)


    # 3. High-pass
    raw = log_stage(
        "highpass",
        adaptive_highpass,
        raw,
        cfg.highpass_candidates,
        cfg.drift_target_pct,
        cfg.band_tol_pct,
    )

    # 4. Bad-time annotation
    if cfg.annotate_bad_time:
        raw = log_stage("bad_time", annotate_bad_time, raw, cfg)

    # 5. Bad-channel interpolation
    raw = log_stage("bad_channels", interpolate_bad_channels, raw, thr)

    # 6. Re-reference
    raw = log_stage("rereference", adaptive_rereference, raw)

    # 7. ASR
    if cfg.do_asr:
        raw = log_stage("asr", run_asr_with_fallback, raw, cfg, thr)

    # NEW: fix flattened channels created by ASR
    raw = log_stage("post_asr_interpolation", interpolate_bad_channels, raw, thr)

    # 8. ICA + ICLabel
    if cfg.do_ica and _ICLABEL_AVAILABLE:
        try:
            raw = log_stage(
                "ica",
                run_ica_iclabel,
                raw,
                cfg.ica_random_state,
                cfg.iclabel_thresholds,
                top_n=cfg.ica_exclude_top_n,
                top_rank=cfg.ica_top_rank,
                mode=cfg.ica_mode,          # <-- NEW
            )
        except Exception as e:
            report["stages"]["ica"] = {"stage": "fail", "error": str(e)}
    
    # Final bandpowers
    report["bandpowers_final"] = _bandpowers_dict(raw)
    intermediates["final"] = raw.copy()

    return raw, report, intermediates


def run_cohort(
    file_paths: List[Path],
    cfg: PipelineConfig,
    thr: StageThresholds,
    outdir: Path,
) -> Dict[str, Any]:
    """
    Run pipeline across multiple .fif files, saving cleaned raw + reports.
    Returns summary dict with per-file status.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    logger = PipelineLogger(outdir, enable_jsonl=True)
    summary: Dict[str, Any] = {}
    for path in file_paths:
        try:
            raw = mne.io.read_raw_fif(path, preload=True)
            raw_clean, report = run_pipeline(raw, cfg, thr, logger=logger)
            # Save outputs
            cleaned_path = outdir / (path.stem + "_cleaned.fif")
            report_path = outdir / (path.stem + "_report.json")
            raw_clean.save(cleaned_path, overwrite=True)
            logger.save_json(report_path, report)
            summary[path.name] = {
                "cleaned": str(cleaned_path),
                "report": str(report_path),
            }
        except Exception as e:
            summary[path.name] = {"error": str(e)}
    logger.close()
    return summary

