# eeg_features.py — industry-grade EEG feature extraction with QC and per-channel controls
from __future__ import annotations

import time
import math
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

# ---- Required deps
try:
    import mne
except Exception as e:
    raise ImportError("This module requires MNE-Python. Please install `mne`.") from e

from scipy.signal import welch, find_peaks, detrend, resample_poly
from scipy.spatial import cKDTree

# ---- Optional deps (graceful fallbacks)
try:
    from mne_connectivity import spectral_connectivity_epochs as _spectral_connectivity_epochs
    _HAS_MNE_CONNECTIVITY = True
    _HAS_MNE_LEGACY_CONNECTIVITY = False
except Exception:
    _HAS_MNE_CONNECTIVITY = False
    try:
        from mne.connectivity import spectral_connectivity as _spectral_connectivity
        _HAS_MNE_LEGACY_CONNECTIVITY = True
    except Exception:
        _HAS_MNE_LEGACY_CONNECTIVITY = False

try:
    import networkx as nx
    _HAS_NETWORKX = True
except Exception:
    _HAS_NETWORKX = False

try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    from numpy.lib.stride_tricks import sliding_window_view
except Exception:
    sliding_window_view = None

try:
    from fooof import FOOOF
    _HAS_FOOOF = True
except Exception:
    _HAS_FOOOF = False


# ------------------------
# Logging
# ------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    # default handler for scripts; frameworks can override
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ------------------------
# Band defs
# ------------------------
@dataclass
class BandDef:
    delta: Tuple[float, float] = (1.0, 4.0)
    theta: Tuple[float, float] = (4.0, 8.0)
    alpha: Tuple[float, float] = (8.0, 13.0)
    beta:  Tuple[float, float] = (13.0, 30.0)
    gamma: Tuple[float, float] = (30.0, 45.0)


# ------------------------
# Utilities (numerics)
# ------------------------
def _band_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    fmin, fmax = band
    return (freqs >= fmin) & (freqs <= fmax)


def _psd_welch(
    data: np.ndarray,
    sfreq: float,
    nperseg: Optional[int] = None,
    fmin: float = 0.5,
    fmax: float = 45.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD per channel. data=(n_ch, n_samples) -> (freqs, psd[n_ch, n_freq])."""
    if nperseg is None:
        nperseg = int(round(2.0 * sfreq))  # 2 s
    freqs, psd = welch(
        data,
        fs=sfreq,
        nperseg=max(8, nperseg),
        noverlap=nperseg // 2,
        detrend="constant",
        axis=-1,
        scaling="density",
    )
    sel = (freqs >= fmin) & (freqs <= fmax)
    return freqs[sel], psd[..., sel]


def _relative_power(psd: np.ndarray) -> np.ndarray:
    denom = np.sum(psd, axis=-1, keepdims=True)
    denom[denom == 0] = np.nan
    return psd / denom


def _safe_mean(x: ArrayLike, axis=None) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(x, axis=axis))


def _center_of_gravity(freqs: np.ndarray, psd1d: np.ndarray) -> float:
    w = np.sum(psd1d)
    if w <= 0 or not np.isfinite(w):
        return np.nan
    return float(np.sum(freqs * psd1d) / w)


def _peak_frequency(freqs: np.ndarray, psd1d: np.ndarray, band: Tuple[float, float]) -> Tuple[float, float]:
    m = _band_mask(freqs, band)
    if not np.any(m):
        return np.nan, np.nan
    idx = int(np.argmax(psd1d[m]))
    return float(freqs[m][idx]), float(psd1d[m][idx])


def _loglog_linear_fit(freqs: np.ndarray, psd1d: np.ndarray, fmin=2.0, fmax=40.0) -> Tuple[float, float]:
    m = (freqs >= fmin) & (freqs <= fmax)
    m &= ~((freqs >= 48.0) & (freqs <= 52.0))  # remove 50 Hz neighborhood
    x = np.log10(freqs[m] + 1e-12)
    y = np.log10(psd1d[m] + np.finfo(float).eps)
    if x.size < 3:
        return np.nan, np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)  # slope, offset


def _embed_signal(x: np.ndarray, m: int) -> np.ndarray:
    if sliding_window_view is not None:
        # shape (N-m+1, m)
        return sliding_window_view(x, m)
    N = len(x) - m + 1
    if N <= 0:
        return np.empty((0, m), float)
    return np.vstack([x[i:i+N] for i in range(m)]).T


def _sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    x = np.asarray(x, float)
    N = len(x)
    if N < m + 2:
        return np.nan
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    tol = r * sd

    def _count_pairs(mdim: int) -> float:
        emb = _embed_signal(x, mdim)
        if emb.shape[0] == 0:
            return 0.0
        tree = cKDTree(emb)
        neigh = tree.query_ball_tree(tree, tol, p=np.inf)
        total_pairs = 0.5 * sum((len(idx) - 1) for idx in neigh)
        return float(total_pairs)

    B = _count_pairs(m)
    A = _count_pairs(m + 1)
    if B == 0.0 or A == 0.0:
        return np.nan
    return float(-np.log(A / B))


def _permutation_entropy(x: np.ndarray, m: int = 3, delay: int = 1) -> float:
    x = np.asarray(x)
    n = len(x) - delay * (m - 1)
    if n <= 0:
        return np.nan
    perms = {}
    for i in range(n):
        pattern = tuple(np.argsort(x[i:i + delay * m:delay]))
        perms[pattern] = perms.get(pattern, 0) + 1
    p = np.array(list(perms.values()), float)
    p /= p.sum()
    denom = math.lgamma(int(m) + 1)  # ln(m!)
    num = -np.sum(p * np.log(p + np.finfo(float).eps))
    return float(num / denom)


def _higuchi_fd(x: np.ndarray, kmax: int = 8) -> float:
    N = len(x)
    if N < (kmax + 1):
        return np.nan
    L = np.zeros(kmax)
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            dist = np.sum(np.abs(np.diff(x[idx])))
            scale = (N - 1) / (len(idx) * k)
            Lm.append(dist * scale)
        L[k - 1] = np.mean(Lm) if Lm else np.nan
    ks = np.arange(1, kmax + 1, dtype=float)
    valid = np.isfinite(L) & (L > 0)
    if valid.sum() < 3:
        return np.nan
    return float(-np.polyfit(np.log(ks[valid]), np.log(L[valid]), 1)[0])


def _dfa_alpha(x: np.ndarray) -> float:
    x = detrend(np.asarray(x, float), type='constant')
    y = np.cumsum(x - np.mean(x))
    N = len(y)
    svals = np.unique(np.logspace(np.log10(4), np.log10(max(8, N // 4)), num=10, dtype=int))
    if len(svals) < 5:
        return np.nan
    F = []
    for s in svals:
        ns = N // s
        if ns < 2:
            continue
        y_ = y[:ns * s].reshape(ns, s)
        t = np.arange(s)
        coeffs = np.polyfit(t, y_.T, 1)
        trend = (coeffs[0][:, None] * t + coeffs[1][:, None])
        F.append(np.sqrt(np.mean((y_ - trend) ** 2)))
    F = np.array(F, float)
    if len(F) < 2:
        return np.nan
    xs = np.log(svals[:len(F)] + 1e-12)
    ys = np.log(F + 1e-12)
    valid = np.isfinite(xs) & np.isfinite(ys)
    if valid.sum() < 2:
        return np.nan
    return float(np.polyfit(xs[valid], ys[valid], 1)[0])


def _hurst(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    N = len(x)
    if N < 128:
        return np.nan
    T = np.arange(10, N // 2, step=max(1, N // 30))
    if len(T) < 5:
        return np.nan
    RS = []
    for t in T:
        segs = N // t
        if segs < 2:
            continue
        X = x[:segs * t].reshape(segs, t)
        Y = X - X.mean(axis=1, keepdims=True)
        Z = np.cumsum(Y, axis=1)
        R = Z.max(axis=1) - Z.min(axis=1)
        S = X.std(axis=1)
        RS.append(np.mean(R / (S + 1e-12)))
    RS = np.array(RS, float)
    if RS.size < 3:
        return np.nan
    xs = np.log(T[:len(RS)] + 1e-12)
    ys = np.log(RS + 1e-12)
    valid = np.isfinite(xs) & np.isfinite(ys)
    if valid.sum() < 2:
        return np.nan
    return float(np.polyfit(xs[valid], ys[valid], 1)[0])


def _sanitize_key(txt: str) -> str:
    # safe feature key suffix from channel name
    return txt.replace(" ", "").replace(".", "").replace("-", "_")

def _lzc_binary(bseq: np.ndarray, normalize: bool = True) -> float:
    """
    Lempel–Ziv (LZ76) complexity for a binary sequence.
    If normalize=True, returns c(n) / (n / log2(n)) in [~0, 1+].
    """
    # ensure 1D bool/int array
    b = np.asarray(bseq).astype(bool).astype(int).tolist()
    n = len(b)
    if n < 2:
        return np.nan

    # Convert to string so we can compare substrings fast
    s = ''.join('1' if x else '0' for x in b)

    # LZ76 incremental parsing (dictionary-based)
    i = 0
    k = 1
    l = 1
    c = 1
    substrings = {s[0]}  # dictionary of seen phrases

    while True:
        if l + k > n:
            c += 1
            break

        fragment = s[i:l + k]  # current phrase

        if fragment in substrings:
            k += 1
            # If we reach the end, count final phrase
            if l + k > n:
                c += 1
                break
        else:
            substrings.add(fragment)
            c += 1
            l = l + k
            i = 0
            k = 1
            if l >= n:
                break

    if not normalize:
        return float(c)

    denom = (n / np.log2(n)) if n > 1 else np.nan
    if not np.isfinite(denom) or denom == 0:
        return np.nan
    return float(c / denom)

def _serialize_matrix_dict(mats: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
    """
    Save numpy arrays (and nested dicts containing arrays) to disk, return JSON-safe references.
    - Arrays -> .npy file + metadata
    - Dicts -> recurse
    - Lists/Tuples -> either values (if scalars) or recurse by index
    """
    base_path.mkdir(parents=True, exist_ok=True)

    def save_item(name: str, obj: Any):
        if isinstance(obj, np.ndarray):
            out_file = base_path / f"{name}.npy"
            np.save(out_file, obj)
            return {"type": "ndarray", "shape": list(obj.shape), "path": str(out_file.resolve())}

        if isinstance(obj, dict):
            return {
                "type": "dict",
                "children": {str(k): save_item(f"{name}_{k}", v) for k, v in obj.items()},
            }

        if isinstance(obj, (list, tuple)):
            # scalar list -> store directly
            if all(isinstance(v, (int, float, bool, str, type(None))) for v in obj):
                return {"type": "list", "values": list(obj)}
            # otherwise recurse
            return {
                "type": "list",
                "children": {str(i): save_item(f"{name}_{i}", v) for i, v in enumerate(obj)},
            }

        # scalar fallback
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    return {str(name): save_item(str(name), arr) for name, arr in mats.items()}


def _embed_signal_delay(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Time-delay embedding: [x[t], x[t+tau], ..., x[t+(m-1)*tau]]
    Returns shape (N_embed, m).
    """
    x = np.asarray(x, float)
    tau = int(max(1, tau))
    m = int(max(2, m))
    N = x.size - (m - 1) * tau
    if N <= 0:
        return np.empty((0, m), float)
    out = np.empty((N, m), float)
    for k in range(m):
        out[:, k] = x[k * tau : k * tau + N]
    return out


def _downsample_for_rqa(x: np.ndarray, sfreq: float, target_fs: float):
    """
    Downsample with anti-alias filtering via resample_poly when possible.
    Returns (x_ds, fs_used). If sfreq <= target_fs, returns original.
    """
    sfreq = float(sfreq)
    target_fs = float(target_fs)
    if not np.isfinite(sfreq) or not np.isfinite(target_fs) or target_fs <= 0:
        return np.asarray(x, float), sfreq
    if sfreq <= target_fs + 1e-9:
        return np.asarray(x, float), sfreq

    ratio = sfreq / target_fs
    down = int(round(ratio))

    # If it matches well (e.g. 1000 -> 200 = /5), do exact integer decimation via polyphase
    if down >= 2 and abs((sfreq / down) - target_fs) / target_fs < 0.02:
        y = resample_poly(np.asarray(x, float), up=1, down=down)
        return y, sfreq / down

    # Fallback approximate
    down = max(2, int(round(sfreq / target_fs)))
    y = resample_poly(np.asarray(x, float), up=1, down=down)
    return y, sfreq / down


# ------------------------
# EEGFeatures 
# ------------------------
class EEGFeatures:
    """
    Industry-grade EEG feature extraction with QC, per-channel controls, and deterministic behavior.

    Parameters
    ----------
    raw : mne.io.BaseRaw | None
    epochs : mne.Epochs | None
    picks : "eeg" | list[int|str]
    per_channel : bool
        If True, export per-channel features (e.g., spectral.rel_alpha_Fp1).
    return_averages : bool
        If True, export channel-averaged features (e.g., spectral.rel_alpha_avg).
    qc : bool
        If True, compute QC meta for each family.
    bands : BandDef
    epoch_len : float
        Length (s) for auto fixed-length epochs when only Raw is provided.
    epoch_overlap : float
        Overlap fraction in [0, 0.9) for auto epochs.
    density : float
        Target density for graph thresholding (0<d<=1).
    random_state : int
        Seed for stochastic parts (k-means).
    complexity_target_fs : float
        Subsample to this FS for complexity metrics (speed).
    complexity_max_seconds : int | None
        Cap duration used per channel for complexity (speed/consistency).
    """

    def __init__(
        self,
        raw: Optional[mne.io.BaseRaw] = None,
        epochs: Optional[mne.Epochs] = None,
        picks: Union[str, List[int], List[str]] = "eeg",
        per_channel: bool = False,
        return_averages: bool = True,
        qc: bool = True,
        bands: BandDef = BandDef(),
        epoch_len: float = 2.0,
        epoch_overlap: float = 0.5,
        density: float = 0.2,
        random_state: int = 42,
        complexity_target_fs: float = 128.0,
        complexity_max_seconds: Optional[int] = 30,
        return_matrices: bool = False,   # <-- NEW
         # ---- RQA / Recurrence controls (memory-safe)
        recurrence_target_fs: float = 200.0,
        recurrence_win_sec: float = 10.0,
        recurrence_overlap: float = 0.5,
        recurrence_m: int = 3,
        recurrence_tau_ms: float = 12.0,
        recurrence_eps: float = 0.1,
        recurrence_det_len: int = 4,
        recurrence_max_points: int = 2000,
        recurrence_max_seconds: Optional[float] = 60.0,
    ):
        if raw is None and epochs is None:
            raise ValueError("Provide at least one of `raw` or `epochs`.")
        self.raw = raw
        self.epochs = epochs
        self.picks = picks
        self.per_channel = bool(per_channel)
        self.return_averages = bool(return_averages)
        self.qc_enabled = bool(qc)
        self.bands = bands
        self.epoch_len = float(epoch_len)
        self.epoch_overlap = float(epoch_overlap)
        self.density = float(density)
        self.random_state = int(random_state)
        self.complexity_target_fs = float(complexity_target_fs)
        self.complexity_max_seconds = complexity_max_seconds
        self.return_matrices = return_matrices 

        # Recurrence settings (only used in features_recurrence)
        self.recurrence_target_fs = float(recurrence_target_fs)
        self.recurrence_win_sec = float(recurrence_win_sec)
        self.recurrence_overlap = float(recurrence_overlap)
        self.recurrence_m = int(recurrence_m)
        self.recurrence_tau_ms = float(recurrence_tau_ms)
        self.recurrence_eps = float(recurrence_eps)
        self.recurrence_det_len = int(recurrence_det_len)
        self.recurrence_max_points = int(recurrence_max_points)
        self.recurrence_max_seconds = recurrence_max_seconds


        if self.sfreq is None or not np.isfinite(self.sfreq):
            raise ValueError("Sampling frequency could not be determined.")

        self._init_channel_info()

        # QC accumulators (filled per family)
        self._qc: Dict[str, Dict[str, Any]] = {}
        self._epochs_cache = None
        # self._fc_cache: Dict[Tuple[str, str, float, float], np.ndarray] = {}
        self._fc_cache: Dict[tuple, np.ndarray] = {}

    # ---------- Core info ----------
    @property
    def sfreq(self) -> Optional[float]:
        if self.epochs is not None:
            return float(self.epochs.info["sfreq"])
        if self.raw is not None:
            return float(self.raw.info["sfreq"])
        return None

    def _get_data_matrix(self) -> np.ndarray:
        if self.raw is not None:
            data = self.raw.get_data(picks=self.pick_idx)
        else:
            data = self.epochs.get_data()[:, self.pick_idx, :]
            data = np.concatenate(data, axis=-1)
        return np.array(data, float)

    def _ensure_epochs(self) -> mne.Epochs:
        """
        Return epochs with the same picked channel set/order as self.ch_names/self.pick_idx.
        If epochs were provided, we still pick to keep channel order consistent.
        If only raw was provided, we create fixed-length epochs once and cache them.
        """
        # If epochs were provided, return a picked copy so channel set/order matches the rest
        if self.epochs is not None:
            ep = self.epochs.copy()
            if self.picks == "eeg":
                ep.pick_types(eeg=True)
            else:
                ep.pick(self.picks)
            # enforce same channel order as initialized
            ep.pick(self.ch_names)
            return ep

        assert self.raw is not None

        # Cache auto-epochs so we don't recreate them repeatedly
        if getattr(self, "_epochs_cache", None) is not None:
            return self._epochs_cache

        dur = float(self.epoch_len)
        step = dur * (1.0 - float(self.epoch_overlap))

        events = mne.make_fixed_length_events(self.raw, id=1, duration=step)
        ep = mne.Epochs(
            self.raw,
            events,
            event_id=1,
            tmin=0.0,
            tmax=dur - (1.0 / self.raw.info["sfreq"]), # tmax=dur,
            baseline=None,
            preload=True,
            reject_by_annotation=True,
        )

        if self.picks == "eeg":
            ep.pick_types(eeg=True)
        else:
            ep.pick(self.picks)

        # enforce same channel order as initialized
        ep.pick(self.ch_names)

        self._epochs_cache = ep
        return ep




    def _init_channel_info(self):
        info = (self.raw.info if self.raw is not None else self.epochs.info)

        if self.picks == "eeg":
            picks = mne.pick_types(info, eeg=True)
        else:
            # Support both List[int] and List[str]
            if isinstance(self.picks, (list, tuple, np.ndarray)) and len(self.picks) > 0 and all(
                isinstance(p, (int, np.integer)) for p in self.picks
            ):
                picks = np.array(self.picks, dtype=int)
                # basic bounds check
                nchs = len(info["ch_names"])
                if np.any(picks < 0) or np.any(picks >= nchs):
                    raise ValueError(f"`picks` contains out-of-range channel indices (0..{nchs-1}).")
            else:
                picks = mne.pick_channels(info["ch_names"], list(self.picks))

        if picks is None or len(picks) == 0:
            raise ValueError("No channels selected after applying `picks`.")

        self.pick_idx = np.array(picks, dtype=int)
        self.ch_names = [info["ch_names"][p] for p in self.pick_idx]

        # positions if available
        xs, ys, zs = [], [], []
        for p in self.pick_idx:
            loc = info["chs"][int(p)]["loc"]
            if loc is not None and np.any(np.isfinite(loc[:3])):
                x, y, z = loc[:3]
            else:
                x = y = z = np.nan
            xs.append(x); ys.append(y); zs.append(z)
        self.pos = np.c_[xs, ys, zs]

        # masks (fallback to name-based if positions missing)
        valid = np.isfinite(self.pos).all(axis=1)
        names = np.array(self.ch_names, str)
        if valid.sum() >= max(8, int(0.5 * len(self.ch_names))):
            x, y = self.pos[:, 0], self.pos[:, 1]
            xm = np.nanmedian(x[valid])
            q25, q75 = np.nanquantile(y[valid], [0.25, 0.75])
            self.left_mask  = valid & (x < xm)
            self.right_mask = valid & (x >= xm)
            self.frontal_mask   = valid & (y >= q75)
            self.occipital_mask = valid & (y <= q25)
        else:
            self.left_mask = np.array([n.startswith(("F1","F3","F5","F7","FT7","T7","TP7","P7","P5","P3","P1","PO7","PO3","O1","C1","C3","C5")) for n in names])
            self.right_mask= np.array([n.startswith(("F2","F4","F6","F8","FT8","T8","TP8","P8","P6","P4","P2","PO8","PO4","O2","C2","C4","C6")) for n in names])
            self.frontal_mask = np.array([n.startswith(("Fp","AF","F")) for n in names])
            self.occipital_mask = np.array([n.startswith(("O","PO")) for n in names])


    # ---------- Summarizers ----------
    def _summarize_vector(self, arr: np.ndarray, family: str, name: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.per_channel:
            for i, ch in enumerate(self.ch_names):
                key = f"{family}.{name}_{_sanitize_key(ch)}"
                out[key] = float(arr[i]) if np.isfinite(arr[i]) else np.nan
        if self.return_averages:
            out[f"{family}.{name}_avg"] = _safe_mean(arr)
        return out

    def _qc_record(self, family: str, t0: float, vecs: List[np.ndarray]):
        if not self.qc_enabled:
            return
        duration = time.time() - t0
        n_ch = len(self.ch_names)
        nan_counts = int(np.sum([np.isnan(v).sum() for v in vecs]))
        n_vals = int(np.sum([v.size for v in vecs]))
        self._qc[family] = {
            "runtime_sec": round(duration, 4),
            "n_channels": int(n_ch),
            "nan_values": int(nan_counts),
            "n_values": int(n_vals),
            "nan_rate_pct": round(100.0 * nan_counts / max(1, n_vals), 3),
        }
        logger.info(f"[QC:{family}] {duration:.2f}s | channels={n_ch} | "
                    f"NaN {nan_counts}/{n_vals} ({self._qc[family]['nan_rate_pct']}%)")

    # ================================
    # Feature families
    # ================================
    def features_spectral(self, fmin: float = 0.5, fmax: float = 45.0) -> Dict[str, float]:
        t0 = time.time()
        data = self._get_data_matrix()
        sf = self.sfreq
        freqs, psd = _psd_welch(data, sf, fmin=fmin, fmax=fmax)
        psd_rel = _relative_power(psd)

        bd = self.bands
        # absolute band powers
        def band_abs(band): 
            m = _band_mask(freqs, band)
            return np.trapz(psd[:, m], freqs[m], axis=-1)
        abs_delta = band_abs(bd.delta)
        abs_theta = band_abs(bd.theta)
        abs_alpha = band_abs(bd.alpha)
        abs_beta  = band_abs(bd.beta)
        abs_gamma = band_abs(bd.gamma)

        # relative band powers
        def band_rel(band):
            m = _band_mask(freqs, band)
            return np.trapz(psd_rel[:, m], freqs[m], axis=-1)
        rel_delta = band_rel(bd.delta)
        rel_theta = band_rel(bd.theta)
        rel_alpha = band_rel(bd.alpha)
        rel_beta  = band_rel(bd.beta)
        rel_gamma = band_rel(bd.gamma)

        # PAF, alpha amplitude, alpha COG, beta peak
        mask_alpha = _band_mask(freqs, bd.alpha)
        cog_alpha = np.array([_center_of_gravity(freqs[mask_alpha], psd[ch, mask_alpha]) for ch in range(psd.shape[0])])
        paf = np.empty(psd.shape[0]); pa_amp = np.empty(psd.shape[0]); bpf = np.empty(psd.shape[0])
        for ch in range(psd.shape[0]):
            f, a = _peak_frequency(freqs, psd[ch], bd.alpha)
            paf[ch] = f; pa_amp[ch] = a
            f2, _ = _peak_frequency(freqs, psd[ch], bd.beta)
            bpf[ch] = f2

        # ratios
        eps = 1e-12
        theta_alpha_ratio = rel_theta / (rel_alpha + eps)
        theta_beta_ratio  = rel_theta / (rel_beta + eps)
        delta_alpha_ratio = rel_delta / (rel_alpha + eps)

        a1 = np.trapz(psd_rel[:, _band_mask(freqs, (8.0, 10.0))], freqs[_band_mask(freqs, (8.0, 10.0))], axis=-1)
        a2 = np.trapz(psd_rel[:, _band_mask(freqs, (10.0,12.0))], freqs[_band_mask(freqs, (10.0,12.0))], axis=-1)
        alpha12_ratio = a1 / (a2 + eps)

        # Frontal alpha asymmetry (Right - Left, relative alpha)
        if np.any(self.left_mask) and np.any(self.right_mask):
            # broadcast group value to channels (for per-channel diagnostic if needed)
            right_mean = _safe_mean(rel_alpha[self.right_mask])
            left_mean  = _safe_mean(rel_alpha[self.left_mask])
            faa_scalar = right_mean - left_mean
        else:
            faa_scalar = np.nan

        # Spectral entropy (per-channel)
        p = np.clip(psd_rel, np.finfo(float).eps, 1.0)
        spec_ent = -np.sum(p * np.log(p), axis=-1) / np.log(p.shape[-1])

        # Alpha power variability across time (relative alpha via sliding Welch)
        nwin = int(round(2.0 * sf))
        step = max(1, nwin // 2)
        alpha_var = np.full(psd.shape[0], np.nan, float)
        for ch in range(data.shape[0]):
            vals = []
            for start in range(0, data.shape[1] - nwin + 1, step):
                seg = data[ch, start:start + nwin]
                f, P = welch(seg, fs=sf, nperseg=min(nwin, len(seg)), noverlap=0, scaling="density")
                m = _band_mask(f, bd.alpha)
                a_area = float(np.trapz(P[m], f[m])) + eps
                t_area = float(np.trapz(P, f)) + eps
                vals.append(a_area / t_area)
            alpha_var[ch] = np.var(vals) if len(vals) > 1 else np.nan

        # Aperiodic slope/offset/knee (per-channel)
        ap_slope = np.full(psd.shape[0], np.nan, float)
        ap_offset = np.full(psd.shape[0], np.nan, float)
        ap_knee = np.full(psd.shape[0], np.nan, float)
        if _HAS_FOOOF:
            f_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.5
            pwl = (max(1.0, 2.0 * f_res), 12.0)
            for ch in range(psd.shape[0]):
                psd_lin = np.maximum(psd[ch], np.finfo(float).tiny)
                try:
                    fm = FOOOF(aperiodic_mode="knee", peak_width_limits=pwl)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fm.fit(freqs, psd_lin, [2, 40])
                    if fm.aperiodic_params_ is None:
                        raise RuntimeError
                    if fm.aperiodic_mode == "knee":
                        offset, knee, exponent = fm.aperiodic_params_
                        ap_offset[ch] = float(offset)
                        ap_knee[ch] = float(knee)
                        ap_slope[ch] = float(-exponent)
                    else:
                        offset, exponent = fm.aperiodic_params_
                        ap_offset[ch] = float(offset)
                        ap_slope[ch] = float(-exponent)
                        ap_knee[ch] = np.nan
                except Exception:
                    a, b = _loglog_linear_fit(freqs, psd_lin)
                    ap_slope[ch] = float(a); ap_offset[ch] = float(b); ap_knee[ch] = np.nan
        else:
            for ch in range(psd.shape[0]):
                a, b = _loglog_linear_fit(freqs, psd[ch])
                ap_slope[ch] = a; ap_offset[ch] = b; ap_knee[ch] = np.nan

        # Alpha posterior–anterior gradient (occipital − frontal, rel alpha)
        if np.any(self.frontal_mask) and np.any(self.occipital_mask):
            alpha_grad_scalar = _safe_mean(rel_alpha[self.occipital_mask]) - _safe_mean(rel_alpha[self.frontal_mask])
        else:
            alpha_grad_scalar = np.nan

        out: Dict[str, float] = {}
        fam = "spectral"
        # abs/rel powers
        out.update(self._summarize_vector(abs_delta, fam, "abs_delta"))
        out.update(self._summarize_vector(abs_theta, fam, "abs_theta"))
        out.update(self._summarize_vector(abs_alpha, fam, "abs_alpha"))
        out.update(self._summarize_vector(abs_beta,  fam, "abs_beta"))
        out.update(self._summarize_vector(abs_gamma, fam, "abs_gamma"))

        out.update(self._summarize_vector(rel_delta, fam, "rel_delta"))
        out.update(self._summarize_vector(rel_theta, fam, "rel_theta"))
        out.update(self._summarize_vector(rel_alpha, fam, "rel_alpha"))
        out.update(self._summarize_vector(rel_beta,  fam, "rel_beta"))
        out.update(self._summarize_vector(rel_gamma, fam, "rel_gamma"))

        # peaks/cog
        out.update(self._summarize_vector(paf, fam, "peak_alpha_frequency"))
        out.update(self._summarize_vector(pa_amp, fam, "alpha_peak_amplitude"))
        out.update(self._summarize_vector(cog_alpha, fam, "alpha_center_of_gravity"))
        out.update(self._summarize_vector(bpf, fam, "beta_peak_frequency"))

        # ratios
        out.update(self._summarize_vector(theta_alpha_ratio, fam, "theta_alpha_ratio"))
        out.update(self._summarize_vector(theta_beta_ratio,  fam, "theta_beta_ratio"))
        out.update(self._summarize_vector(delta_alpha_ratio, fam, "delta_alpha_ratio"))
        out.update(self._summarize_vector(alpha12_ratio, fam, "alpha1_alpha2_ratio"))

        # entropy & alpha variance
        out.update(self._summarize_vector(spec_ent, fam, "spectral_entropy"))
        out.update(self._summarize_vector(alpha_var, fam, "alpha_power_variability"))

        # aperiodic
        out.update(self._summarize_vector(ap_slope,  fam, "aperiodic_slope"))
        out.update(self._summarize_vector(ap_offset, fam, "aperiodic_offset"))
        out.update(self._summarize_vector(ap_knee,   fam, "aperiodic_knee"))

        # FAA & posterior-anterior gradient (scalars)
        if self.return_averages:
            out[f"{fam}.frontal_alpha_asymmetry_avg"] = float(faa_scalar)
            out[f"{fam}.alpha_posterior_anterior_gradient_avg"] = float(alpha_grad_scalar)

        self._qc_record("spectral", t0, [
            abs_delta, rel_alpha, paf, spec_ent, ap_slope
        ])
        return out

    def _compute_fc(
        self,
        band: str = "alpha",
        method: str = "wpli",
        fmin: Optional[float] = None,
        fmax: Optional[float] = None
    ) -> np.ndarray:
        ep = self._ensure_epochs()
        sf = float(ep.info["sfreq"])

        bd = getattr(self.bands, band)
        fmin = float(bd[0] if fmin is None else fmin)
        fmax = float(bd[1] if fmax is None else fmax)

        methods_map = {"imcoh": "imcoh", "pli": "pli", "wpli": "wpli", "plv": "plv", "aec": "aec"}
        if method not in methods_map:
            raise ValueError(f"Unknown connectivity method {method}.")

        # --- cache (speed + deterministic repeated calls)
        key = (band, method, fmin, fmax, tuple(self.ch_names), float(sf))
        if not hasattr(self, "_fc_cache"):
            self._fc_cache = {}
        if key in self._fc_cache:
            return self._fc_cache[key].copy()

        # IMPORTANT: enforce same channel set + order as the rest of the pipeline
        ep_use = ep.copy().pick(self.ch_names)

        if _HAS_MNE_CONNECTIVITY:
            con = _spectral_connectivity_epochs(
                ep_use,
                method=methods_map[method],
                mode="multitaper",
                sfreq=sf,
                fmin=fmin,
                fmax=fmax,
                faverage=True,
                mt_adaptive=False,
                n_jobs=1,
            )
            mat = con.get_data(output="dense")
        elif _HAS_MNE_LEGACY_CONNECTIVITY:
            data = ep_use.get_data()
            con = _spectral_connectivity(
                data,
                method=methods_map[method],
                mode="multitaper",
                sfreq=sf,
                fmin=fmin,
                fmax=fmax,
                faverage=True,
                mt_adaptive=False,
                n_jobs=1,
            )
            mat = np.squeeze(con[0])
        else:
            raise ImportError("Connectivity requires `mne_connectivity` or legacy `mne.connectivity`.")

        mat = np.squeeze(mat)
        mat = 0.5 * (mat + mat.T)  # symmetrize
        np.fill_diagonal(mat, 0.0)

        self._fc_cache[key] = mat
        return mat.copy()



    def _threshold_fc(self, mat: np.ndarray, density: Optional[float] = None) -> np.ndarray:
        dens = self.density if density is None else float(density)
        if not (0 < dens <= 1):
            raise ValueError("density must be in (0, 1].")
        if dens == 1.0:
            return mat.copy()
        n = mat.shape[0]
        iu = np.triu_indices(n, k=1)
        vals = np.abs(mat[iu])
        k = max(1, int(round(dens * len(vals))))
        thr = np.partition(vals, -k)[-k]
        out = mat.copy()
        out[np.abs(out) < thr] = 0.0
        np.fill_diagonal(out, 0.0)
        return out

    def _graph_metrics(self, W: np.ndarray) -> Dict[str, float]:
        keys = [
            "global_efficiency", "local_efficiency", "char_path_length",
            "clustering", "small_world_sigma", "modularity", "assortativity",
            "transitivity", "avg_strength", "betweenness", "eigenvector",
            "participation", "rich_club", "hub_disruption_index"
        ]

        if not _HAS_NETWORKX:
            return {k: np.nan for k in keys}

        # ensure numeric, finite-ish
        W = np.asarray(W, float)
        if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] < 2:
            return {k: np.nan for k in keys}

        # networkx builds edges for any non-zero entry; keep only finite weights
        W2 = W.copy()
        W2[~np.isfinite(W2)] = 0.0
        np.fill_diagonal(W2, 0.0)

        G = nx.from_numpy_array(W2)
        if G.number_of_nodes() == 0:
            return {k: np.nan for k in keys}

        # take largest connected component (graph metrics are more stable there)
        comps = list(nx.connected_components(G))
        Gc = G.subgraph(max(comps, key=len)).copy() if comps else G
        if Gc.number_of_nodes() < 2:
            return {k: np.nan for k in keys}

        # distance = 1/weight for path-based metrics
        for u, v, d in Gc.edges(data=True):
            w = float(d.get("weight", 0.0))
            d["distance"] = (1.0 / (w + 1e-12)) if w > 0 else np.inf

        try:
            geff = nx.global_efficiency(Gc)
        except Exception:
            geff = np.nan

        try:
            leff = nx.local_efficiency(Gc) if hasattr(nx, "local_efficiency") else np.nan
        except Exception:
            leff = np.nan

        # characteristic path length (weighted)
        try:
            lengths = dict(nx.all_pairs_dijkstra_path_length(Gc, weight="distance"))
            nodes = list(Gc.nodes())
            L = []
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    lij = lengths[nodes[i]].get(nodes[j], np.inf)
                    if np.isfinite(lij):
                        L.append(lij)
            cpl = float(np.mean(L)) if L else np.nan
        except Exception:
            cpl = np.nan

        try:
            clust = nx.average_clustering(Gc, weight="weight")
        except Exception:
            clust = np.nan

        # small-worldness vs ER baseline (make p available to later blocks!)
        p = np.nan
        sigma = np.nan
        try:
            n = int(Gc.number_of_nodes())
            m = int(Gc.number_of_edges())
            p = 2.0 * m / (n * (n - 1) + 1e-12)

            Gr = nx.erdos_renyi_graph(n, p, seed=self.random_state)
            Cr = nx.average_clustering(Gr)

            try:
                Lr = nx.average_shortest_path_length(Gr)
            except Exception:
                comps_r = list(nx.connected_components(Gr))
                if comps_r:
                    Ggr = Gr.subgraph(max(comps_r, key=len)).copy()
                    Lr = nx.average_shortest_path_length(Ggr)
                else:
                    Lr = np.nan

            sigma = (clust / (Cr + 1e-12)) / (cpl / (Lr + 1e-12))
        except Exception:
            sigma = np.nan

        # modularity + participation coefficient
        try:
            from networkx.algorithms.community import greedy_modularity_communities, modularity

            comms = list(greedy_modularity_communities(Gc, weight="weight"))
            Q = modularity(Gc, comms, weight="weight") if comms else np.nan

            pc = []
            for n_i in Gc.nodes():
                k_i = sum(Gc[n_i][j].get("weight", 0.0) for j in Gc.neighbors(n_i))
                if k_i <= 0:
                    pc.append(np.nan)
                    continue
                sum_sq = 0.0
                for c in comms:
                    w_i_c = sum(Gc[n_i][j].get("weight", 0.0) for j in Gc.neighbors(n_i) if j in c)
                    sum_sq += (w_i_c / k_i) ** 2
                pc.append(1.0 - sum_sq)

            participation = float(np.nanmean(pc))
        except Exception:
            Q = np.nan
            participation = np.nan

        try:
            assort = nx.degree_assortativity_coefficient(Gc, weight="weight")
        except Exception:
            assort = np.nan

        try:
            trans = nx.transitivity(Gc)
        except Exception:
            trans = np.nan

        try:
            strength = float(np.mean([d for _, d in Gc.degree(weight="weight")]))
        except Exception:
            strength = np.nan

        try:
            btw = float(np.mean(list(nx.betweenness_centrality(Gc, weight="distance").values())))
        except Exception:
            btw = np.nan

        try:
            eig = float(np.mean(list(nx.eigenvector_centrality_numpy(Gc, weight="weight").values())))
        except Exception:
            eig = np.nan

        # rich-club (degree-based)
        try:
            degs = np.array([d for _, d in Gc.degree()], dtype=float)
            if degs.size:
                k_thr = int(np.quantile(degs, 0.9))
                rc = nx.rich_club_coefficient(Gc, normalized=False).get(k_thr, np.nan)
            else:
                rc = np.nan
        except Exception:
            rc = np.nan

        # hub disruption index (rough): compare eigenvector centrality distribution vs ER baseline
        try:
            if not np.isfinite(p):
                hub_disruption = np.nan
            else:
                ec = nx.eigenvector_centrality_numpy(Gc, weight="weight")
                ec_vals = np.array(list(ec.values()), dtype=float)

                Gr = nx.erdos_renyi_graph(int(Gc.number_of_nodes()), float(p), seed=self.random_state)
                if nx.is_connected(Gr):
                    ec_r = nx.eigenvector_centrality_numpy(Gr)
                    ec_r_vals = np.array(list(ec_r.values()), dtype=float)

                    hub_disruption = float(np.mean(
                        np.sort(ec_vals) - np.interp(
                            np.linspace(0, 1, ec_vals.size),
                            np.linspace(0, 1, ec_r_vals.size),
                            np.sort(ec_r_vals),
                        )
                    ))
                else:
                    hub_disruption = np.nan
        except Exception:
            hub_disruption = np.nan

        return {
            "global_efficiency": float(geff),
            "local_efficiency": float(leff),
            "char_path_length": float(cpl),
            "clustering": float(clust),
            "small_world_sigma": float(sigma),
            "modularity": float(Q),
            "assortativity": float(assort),
            "transitivity": float(trans),
            "avg_strength": float(strength),
            "betweenness": float(btw),
            "eigenvector": float(eig),
            "participation": float(participation),
            "rich_club": float(rc),
            "hub_disruption_index": float(hub_disruption),
        }


    def features_connectivity_graph(self, band: str = "alpha") -> Dict[str, float]:
        t0 = time.time()
        out: Dict[str, float] = {}

        # Connectivity means from MNE connectivity
        # IMPORTANT: use a dedicated namespace to avoid collisions with envelope-AEC
        for method in ["imcoh", "wpli", "plv", "aec"]:
            try:
                mat = self._compute_fc(band=band, method=method)
                iu = np.triu_indices_from(mat, k=1)
                val = float(np.nanmean(mat[iu])) if mat.size else np.nan
            except Exception:
                val = np.nan

            out[f"connectivity_mne.{method}_mean_{band}"] = val


        # Graph metrics from wPLI
        try:
            fc = self._compute_fc(band=band, method="wpli")
            if self.return_matrices:
                self._last_conn = {"band": band, "method": "wpli", "matrix": fc}
            fc_thr = self._threshold_fc(fc, density=self.density)
            gm = self._graph_metrics(fc_thr)
            for k, v in gm.items():
                out[f"graph.{k}_{band}"] = float(v)
        except Exception:
            pass

        # QC
        self._qc_record("connectivity_graph", t0, [np.array(list(out.values()), float)])
        return out


    # ---------- Microstates ----------
    def _microstate_fit(self, n_classes: int = 4, peak_dist_ms: float = 10.0) -> Dict[str, np.ndarray]:
        if not _HAS_SKLEARN:
            raise ImportError("Microstates require scikit-learn (KMeans).")
        data = self._get_data_matrix()
        x = data - data.mean(axis=1, keepdims=True)
        gfp = np.std(x, axis=0)
        distance = int(round((peak_dist_ms / 1000.0) * self.sfreq))
        peaks, _ = find_peaks(gfp, distance=max(1, distance))
        if len(peaks) < max(50, n_classes * 20):
            warnings.warn("Few GFP peaks; microstate stability may be poor.")
        maps = x[:, peaks]
        maps = maps / (np.linalg.norm(maps, axis=0, keepdims=True) + 1e-12)
        km = KMeans(n_clusters=n_classes, random_state=self.random_state, n_init=10)
        labels_peak = km.fit_predict(np.abs(maps.T))
        proto = []
        for k in range(n_classes):
            idx = np.where(labels_peak == k)[0]
            avg = np.mean(maps[:, idx], axis=1) if len(idx) else np.zeros(maps.shape[0])
            avg = avg / (np.linalg.norm(avg) + 1e-12)
            proto.append(avg)
        proto = np.array(proto)
        x_norm = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-12)
        corr = proto @ x_norm
        labels = np.argmax(np.abs(corr), axis=0)
        return {"maps": proto, "labels": labels, "gfp": gfp}

    def features_microstates(self, n_classes: int = 4) -> Dict[str, float]:
        t0 = time.time()
        ms = self._microstate_fit(n_classes=n_classes)
        labels = ms["labels"]; gfp = ms["gfp"]; sf = self.sfreq
        # runs
        durations = []
        occurrences = 0
        trans_mat = np.zeros((n_classes, n_classes), int)
        last = labels[0]; run_len = 1
        for t in range(1, len(labels)):
            if labels[t] == last:
                run_len += 1
            else:
                durations.append(run_len / sf)
                occurrences += 1
                trans_mat[last, labels[t]] += 1
                last = labels[t]; run_len = 1
        durations.append(run_len / sf); occurrences += 1

        # per class coverage/mean duration/occurrence rate
        coverage = []; mean_dur_class = []; occ_rate_class = []
        for k in range(n_classes):
            idx = np.where(labels == k)[0]
            coverage.append(len(idx) / len(labels))
            runs = []
            rl = 0
            for t in range(len(labels)):
                if labels[t] == k: rl += 1
                elif rl > 0: runs.append(rl / sf); rl = 0
            if rl > 0: runs.append(rl / sf)
            mean_dur_class.append(np.mean(runs) if runs else np.nan)
            occ_rate_class.append(len(runs) / (len(labels) / sf))
        coverage = np.array(coverage); mean_dur_class = np.array(mean_dur_class); occ_rate_class = np.array(occ_rate_class)

        # GFP peak rate
        peaks, _ = find_peaks(gfp, distance=int(round(0.01 * sf)))
        gfp_rate = len(peaks) / (len(gfp) / sf)

        # Transition entropy (weighted by row usage)
        P = trans_mat.astype(float)
        row_sums = P.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            Pn = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums != 0)
        def _row_ent(p):
            z = p[p > 0]
            return -np.sum(z * np.log2(z)) if z.size else 0.0
        row_ent = np.apply_along_axis(_row_ent, 1, Pn)
        weights = row_sums.squeeze()
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        transition_entropy = float(np.sum(row_ent * weights))

        # normalized transition entropy
        n_out = (Pn > 0).sum(axis=1)
        max_row_ent = np.log2(np.clip(n_out, 1, None))
        with np.errstate(divide='ignore', invalid='ignore'):
            row_ent_norm = np.divide(row_ent, max_row_ent, out=np.zeros_like(row_ent), where=max_row_ent > 0)
        transition_entropy_norm = float(np.sum(row_ent_norm * weights))

        # GEV
        proto = ms["maps"]
        data = self._get_data_matrix()
        x = data - data.mean(axis=1, keepdims=True)
        x_norm = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-12)
        corr = proto @ x_norm
        corr_abs_max = np.max(np.abs(corr), axis=0)
        denom = np.sum(gfp ** 2) + 1e-12
        gev = float(np.sum((gfp * corr_abs_max) ** 2) / denom)

        letters = [chr(65 + i) for i in range(n_classes)]
        out: Dict[str, float] = {}
        for k in range(n_classes):
            out[f"microstates.mean_duration_{letters[k]}"] = float(mean_dur_class[k])
            out[f"microstates.coverage_{letters[k]}"] = float(coverage[k])
            out[f"microstates.occurrence_rate_{letters[k]}"] = float(occ_rate_class[k])
        out["microstates.gfp_peak_rate"] = float(gfp_rate)
        out["microstates.gev"] = float(gev)
        out["microstates.transition_entropy"] = float(transition_entropy)
        out["microstates.transition_entropy_norm"] = float(transition_entropy_norm)

        self._qc_record("microstates", t0, [coverage, mean_dur_class])
        return out

    # ---------- Complexity ----------
    def features_complexity(
        self,
        sample_m: int = 2,
        sample_r: float = 0.2,
        mse_scales: Union[List[int], range] = range(2, 11),
        pe_m: int = 3,
        pe_delay: int = 1,
        higuchi_kmax: int = 8,
        lz_binarize: str = "median",
    ) -> Dict[str, float]:
        t0 = time.time()
        x = self._get_data_matrix()
        n_ch = x.shape[0]

        sampen = np.full(n_ch, np.nan)
        mse_auc = np.full(n_ch, np.nan)
        perment = np.full(n_ch, np.nan)
        hfd = np.full(n_ch, np.nan)
        dfa = np.full(n_ch, np.nan)
        hurst = np.full(n_ch, np.nan)
        lzc = np.full(n_ch, np.nan)

        scales = list(mse_scales)

        for ch in range(n_ch):
            sig = np.asarray(x[ch], float)
            if not np.isfinite(sig).any() or sig.size < 256:
                continue

            # decimation for speed
            fs_used = self.sfreq
            if fs_used and self.complexity_target_fs and fs_used > self.complexity_target_fs:
                dec = max(1, int(round(fs_used / self.complexity_target_fs)))
                sig = sig[::dec]
                fs_used = fs_used / dec

            if self.complexity_max_seconds:
                max_pts = int(self.complexity_max_seconds * (fs_used if fs_used else 1.0))
                if sig.size > max_pts:
                    sig = sig[:max_pts]

            sampen[ch] = _sample_entropy(sig, m=sample_m, r=sample_r)

            # MSE (AUC across scales)
            mse_vals = []
            for s in scales:
                if s < 2:
                    mse_vals.append(np.nan); continue
                n = (len(sig) // s) * s
                if n < 2 * (sample_m + 1):
                    mse_vals.append(np.nan); continue
                coarse = sig[:n].reshape(-1, s).mean(axis=1)
                mse_vals.append(_sample_entropy(coarse, m=sample_m, r=sample_r))
            mse_vals = np.array(mse_vals, float)
            if np.isfinite(mse_vals).sum() >= 2:
                idx = np.arange(1, len(mse_vals) + 1, dtype=float)
                mse_auc[ch] = float(np.trapz(np.nan_to_num(mse_vals, nan=0.0), idx) /
                                    (idx[-1] - idx[0] + 1e-12))

            perment[ch] = _permutation_entropy(sig, m=pe_m, delay=pe_delay)
            hfd[ch] = _higuchi_fd(sig, kmax=higuchi_kmax)
            dfa[ch] = _dfa_alpha(sig)
            hurst[ch] = _hurst(sig)

            thr = (np.median(sig) if lz_binarize == "median" else np.mean(sig))
            b = (sig > thr).astype(int)
            # Optional: if sequence is all 0s or 1s after binarization, jitter the threshold slightly
            if b.sum() == 0 or b.sum() == b.size:
                thr = thr + 1e-12 * np.std(sig)  # tiny nudge
                b = (sig > thr).astype(int)

            lzc[ch] = _lzc_binary(b, normalize=True)

        fam = "complexity"
        out: Dict[str, float] = {}
        out.update(self._summarize_vector(sampen,  fam, "sample_entropy"))
        out.update(self._summarize_vector(mse_auc,  fam, "multiscale_entropy_auc"))
        out.update(self._summarize_vector(perment,  fam, "permutation_entropy"))
        out.update(self._summarize_vector(hfd,     fam, "higuchi_fd"))
        out.update(self._summarize_vector(dfa,     fam, "dfa_alpha"))
        out.update(self._summarize_vector(hurst,   fam, "hurst_exponent"))
        out.update(self._summarize_vector(lzc,     fam, "lzc_broadband"))

        self._qc_record("complexity", t0, [sampen, mse_auc, perment, hfd, dfa, hurst, lzc])
        return out

    # ---------- Topography & Asymmetry ----------
    def features_topography_asymmetry(self, connectivity_method: str = "wpli") -> Dict[str, float]:
        t0 = time.time()
        data = self._get_data_matrix()
        sf = self.sfreq
        freqs, psd = _psd_welch(data, sf, fmin=0.5, fmax=45.0)
        psd_rel = _relative_power(psd)
        bd = self.bands

        def band_rel(band):
            m = _band_mask(freqs, band)
            return np.trapz(psd_rel[:, m], freqs[m], axis=-1)

        rel_alpha = band_rel(bd.alpha)
        rel_theta = band_rel(bd.theta)
        rel_beta  = band_rel(bd.beta)
        rel_delta = band_rel(bd.delta)
        rel_gamma = band_rel(bd.gamma)

        valid = np.isfinite(self.pos).all(axis=1)
        y = self.pos[:, 1]
        q10, q35 = np.nanquantile(y[valid], [0.10, 0.35]) if valid.any() else (np.nan, np.nan)
        parietal = valid & (y <= q35) & (y > q10) & (~self.occipital_mask)

        if np.any(self.frontal_mask) and np.any(self.occipital_mask):
            alpha_grad = _safe_mean(rel_alpha[self.occipital_mask]) - _safe_mean(rel_alpha[self.frontal_mask])
        else:
            alpha_grad = np.nan

        asym_vals = []
        if np.any(self.left_mask) and np.any(self.right_mask):
            for rb in [rel_delta, rel_theta, rel_alpha, rel_beta, rel_gamma]:
                diff = abs(_safe_mean(rb[self.left_mask]) - _safe_mean(rb[self.right_mask]))
                asym_vals.append(diff)
            interhemi_asym = float(np.nanmean(asym_vals)) if asym_vals else np.nan
        else:
            interhemi_asym = np.nan

        try:
            # theta fronto-parietal
            fc_theta = self._compute_fc(band="theta", method=connectivity_method)
            idx_f = np.where(self.frontal_mask)[0]
            idx_p = np.where(parietal)[0]
            fp_vals = []
            for i in idx_f:
                for j in idx_p:
                    if i != j: fp_vals.append(fc_theta[i, j])
            fp_mean = float(np.nanmean(fp_vals)) if fp_vals else np.nan

            # alpha occipital-occipital
            fc_alpha = self._compute_fc(band="alpha", method=connectivity_method)
            idx_o = np.where(self.occipital_mask)[0]
            oo_vals = []
            for a in range(len(idx_o)):
                for b in range(a + 1, len(idx_o)):
                    i = idx_o[a]; j = idx_o[b]
                    oo_vals.append(fc_alpha[i, j])
            oo_mean = float(np.nanmean(oo_vals)) if oo_vals else np.nan

            fp_over_oo = (fp_mean / (oo_mean + 1e-12)) if (np.isfinite(fp_mean) and np.isfinite(oo_mean)) else np.nan
        except Exception:
            fp_over_oo = np.nan

        out = {
            "topography.alpha_posterior_anterior_gradient": float(alpha_grad),
            "topography.interhemispheric_power_asymmetry": float(interhemi_asym),
            "topography.fronto_parietal_theta_vs_occipital_alpha_coupling": float(fp_over_oo),
        }
        self._qc_record("topography_asymmetry", t0, [rel_alpha])
        return out


    
    # ---------- Cross-Frequency Coupling ----------

    def _bandpass_filter(self, sig: np.ndarray, band: Tuple[float, float], sf: float) -> np.ndarray:
        """Butterworth bandpass filter, guarding band edges."""
        from scipy.signal import butter, filtfilt
        low, high = float(band[0]), float(band[1])
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= 0:
            return sig  # fallback: no filtering if band is invalid
        if high <= low:
            high = low * 1.25  # tiny widening if inverted/degenerate
        nyq = sf / 2.0
        # normalize & clip into (0, 1)
        wn = [max(1e-6, min(0.999, low / nyq)), max(1e-6, min(0.999, high / nyq))]
        if wn[1] <= wn[0]:  # still degenerate after clipping
            wn[1] = min(0.999, wn[0] * 1.25)
        b, a = butter(4, wn, btype="band")
        return filtfilt(b, a, sig, method="gust")


    def _phase_amplitude_coupling(self, data: np.ndarray, sf: float,
                                  phase_band: Tuple[float, float],
                                  amp_band: Tuple[float, float]) -> np.ndarray:
        """Compute per-channel Phase-Amplitude Coupling (PAC) using Modulation Index (Tort et al.)."""
        from scipy.signal import hilbert

        n_ch = data.shape[0]
        pac_vals = np.zeros(n_ch)

        for ch in range(n_ch):
            sig = data[ch]

            # Phase
            phase_sig = self._bandpass_filter(sig, phase_band, sf)
            phase = np.angle(hilbert(phase_sig))

            # Amplitude
            amp_sig = self._bandpass_filter(sig, amp_band, sf)
            amp = np.abs(hilbert(amp_sig))

            # Bin amplitude by phase
            nbins = 18
            bins = np.linspace(-np.pi, np.pi, nbins + 1)
            digitized = np.digitize(phase, bins) - 1
            mean_amp = np.array([amp[digitized == k].mean() if np.any(digitized == k) else 0
                                 for k in range(nbins)])
            mean_amp /= mean_amp.sum() + 1e-12

            # Compute Modulation Index (KL divergence from uniform)
            H = -np.sum(mean_amp * np.log(mean_amp + 1e-12))
            Hmax = np.log(nbins)
            pac_vals[ch] = (Hmax - H) / Hmax  # normalized MI

        return pac_vals

    def features_cfc(self) -> Dict[str, float]:
        """Cross-Frequency Coupling (PAC + cross-frequency coherence)."""
        from scipy.signal import coherence
        t0 = time.time()
        data = self._get_data_matrix()
        sf = self.sfreq

        out: Dict[str, float] = {}

        # --- PAC features ---
        pac_theta_gamma = self._phase_amplitude_coupling(data, sf, self.bands.theta, self.bands.gamma)
        out.update(self._summarize_vector(pac_theta_gamma, "cfc", "pac_theta_gamma"))

        pac_alpha_beta = self._phase_amplitude_coupling(data, sf, self.bands.alpha, self.bands.beta)
        out.update(self._summarize_vector(pac_alpha_beta, "cfc", "pac_alpha_beta"))

        pac_delta_theta = self._phase_amplitude_coupling(data, sf, self.bands.delta, self.bands.theta)
        out.update(self._summarize_vector(pac_delta_theta, "cfc", "pac_delta_theta"))

        pac_beta_gamma = self._phase_amplitude_coupling(data, sf, self.bands.beta, self.bands.gamma)
        out.update(self._summarize_vector(pac_beta_gamma, "cfc", "pac_beta_gamma"))

        # --- Cross-frequency coherence ---
        def _band_coherence(sig, band1, band2):
            sig1 = self._bandpass_filter(sig, band1, sf)
            sig2 = self._bandpass_filter(sig, band2, sf)
            f, Cxy = coherence(sig1, sig2, fs=sf, nperseg=int(2*sf))
            mask = (f >= min(band1[0], band2[0])) & (f <= max(band1[1], band2[1]))
            return np.nanmean(Cxy[mask]) if np.any(mask) else np.nan

        n_ch = data.shape[0]
        coh_alpha_beta = np.array([_band_coherence(data[ch], self.bands.alpha, self.bands.beta) for ch in range(n_ch)])
        out.update(self._summarize_vector(coh_alpha_beta, "cfc", "coherence_alpha_beta"))

        coh_theta_gamma = np.array([_band_coherence(data[ch], self.bands.theta, self.bands.gamma) for ch in range(n_ch)])
        out.update(self._summarize_vector(coh_theta_gamma, "cfc", "coherence_theta_gamma"))

        # QC
        self._qc_record("cfc", t0, [pac_theta_gamma, pac_alpha_beta, pac_delta_theta,
                                    pac_beta_gamma, coh_alpha_beta, coh_theta_gamma])
        
        if self.return_matrices:
            self._last_cfc = {
                "pac": {
                    "theta_gamma": pac_theta_gamma,
                    "alpha_beta":  pac_alpha_beta,
                    "delta_theta": pac_delta_theta,
                    "beta_gamma":  pac_beta_gamma,
                },
                "coherence": {
                    "alpha_beta":  coh_alpha_beta,
                    "theta_gamma": coh_theta_gamma,
                }
            }
        return out




    # ---------- EO/EC alpha reactivity helper ----------
    def alpha_reactivity(self, other: "EEGFeatures") -> float:
        """Occipital alpha reactivity: (EC rel-alpha) − (EO rel-alpha)."""
        def _rel_alpha(feats: "EEGFeatures") -> Tuple[np.ndarray, np.ndarray]:
            d = feats._get_data_matrix()
            f, P = _psd_welch(d, feats.sfreq, fmin=0.5, fmax=45.0)
            Pr = _relative_power(P)
            m = _band_mask(f, feats.bands.alpha)
            ra = np.trapz(Pr[:, m], f[m], axis=-1)
            return ra, feats.occipital_mask
        a1, o1 = _rel_alpha(self)
        a2, o2 = _rel_alpha(other)
        if not (np.any(o1) and np.any(o2)):
            return float("nan")
        return float(_safe_mean(a2[o2]) - _safe_mean(a1[o1]))



    def features_amplitude_envelope_connectivity(self) -> Dict[str, float]:
        """
        Bandwise amplitude envelope correlation and global synchronization index.
        """
        from scipy.signal import hilbert
        t0 = time.time()
        data = self._get_data_matrix()
        sf = self.sfreq
        bd = self.bands

        out: Dict[str, float] = {}
        for band_name in ["theta", "alpha", "beta", "gamma"]:
            band = getattr(bd, band_name)
            sig_band = np.array([self._bandpass_filter(ch, band, sf) for ch in data])
            amp = np.abs(hilbert(sig_band))

            # Stabilize correlation: z-score per channel so constant/near-constant channels don't explode corrcoef
            amp = amp - np.nanmean(amp, axis=1, keepdims=True)
            std = np.nanstd(amp, axis=1, keepdims=True)
            amp = np.divide(amp, std, out=np.zeros_like(amp), where=std > 1e-12)

            corr = np.corrcoef(amp)
            corr[~np.isfinite(corr)] = np.nan
            iu = np.triu_indices_from(corr, k=1)

            out[f"aec_envelope.mean_{band_name}"] = float(np.nanmean(corr[iu]))
            out[f"aec_envelope.global_sync_{band_name}"] = float(np.nanmean(np.abs(corr[iu])))

        self._qc_record("amplitude_envelope_connectivity", t0, [np.array(list(out.values()), float)])
        return out



    # --- inside class EEGFeatures ---

    # @staticmethod
    # def _recurrence_features(x: np.ndarray, m: int = 3, tau: int = 4, eps: float = 0.1) -> Dict[str, float]:
    #     """Compute recurrence rate and determinism (simple RQA)."""
    #     X = _embed_signal(x, m)
    #     if X.size == 0:
    #         return {"recurrence_rate": np.nan, "determinism": np.nan}
    #     from scipy.spatial.distance import cdist
    #     D = cdist(X, X, metric="euclidean")

    #     thr = np.nanpercentile(D, eps * 100.0)
    #     R = (D < thr).astype(int)

    #     # Recurrence Rate
    #     rr = float(R.sum()) / float(R.size + 1e-12)

    #     # Determinism: fraction of recurrence points in diagonal lines of length >= tau
    #     # (very lightweight proxy)
    #     det_count = 0.0
    #     total_rec = float(R.sum()) + 1e-12
    #     ones_k = np.ones(tau, dtype=int)
    #     for row in R:
    #         # count positions where at least tau consecutive ones occur
    #         det_count += float(np.sum(np.convolve(row, ones_k, mode="valid") >= tau))
    #     det = float(det_count / total_rec)

    #     return {"recurrence_rate": rr, "determinism": det}

    @staticmethod
    def _recurrence_features_windowed(
        x: np.ndarray,
        sfreq: float,
        *,
        m: int = 3,
        tau_ms: float = 12.0,
        eps: float = 0.1,
        det_len: int = 4,
        target_fs: float = 200.0,
        win_sec: float = 10.0,
        overlap: float = 0.5,
        max_points: int = 2000,
        max_seconds: Optional[float] = 60.0,
    ) -> Dict[str, float]:
        """
        Memory-safe RQA:
          - downsample to target_fs (anti-alias) for recurrence only
          - compute per window to keep cdist small
          - tau defined in milliseconds (stable across sampling rates)
          - average RR/DET across windows
        """
        x = np.asarray(x, float)
        if x.size < 256 or not np.isfinite(x).any():
            return {"recurrence_rate": np.nan, "determinism": np.nan}

        # optional duration cap
        if max_seconds is not None and np.isfinite(max_seconds) and max_seconds > 0 and np.isfinite(sfreq):
            max_pts = int(round(float(max_seconds) * float(sfreq)))
            if max_pts > 0 and x.size > max_pts:
                x = x[:max_pts]

        # downsample recurrence only
        x_ds, fs_used = _downsample_for_rqa(x, sfreq=float(sfreq), target_fs=float(target_fs))
        if x_ds.size < 256 or not np.isfinite(x_ds).any():
            return {"recurrence_rate": np.nan, "determinism": np.nan}

        # tau in samples from ms
        tau_samp = int(max(1, round((float(tau_ms) / 1000.0) * float(fs_used))))
        m = int(max(2, m))
        det_len = int(max(2, det_len))

        nwin = int(max(128, round(float(win_sec) * float(fs_used))))
        step = int(max(1, round(nwin * (1.0 - float(overlap)))))

        if x_ds.size < nwin:
            starts = [0]
        else:
            starts = list(range(0, x_ds.size - nwin + 1, step))

        from scipy.spatial.distance import cdist

        rr_list = []
        det_list = []

        for s0 in starts:
            seg = x_ds[s0 : s0 + nwin]
            seg = seg - np.nanmean(seg)
            sd = np.nanstd(seg)
            if not np.isfinite(sd) or sd == 0:
                continue
            seg = seg / sd

            X = _embed_signal_delay(seg, m=m, tau=tau_samp)
            if X.shape[0] < 64:
                continue

            # bound cdist size
            if X.shape[0] > max_points:
                idx = np.linspace(0, X.shape[0] - 1, int(max_points)).astype(int)
                X = X[idx]

            D = cdist(X, X, metric="euclidean")
            thr = np.nanpercentile(D, float(eps) * 100.0)
            if not np.isfinite(thr) or thr <= 0:
                continue

            R = (D < thr).astype(np.int8)
            rr = float(R.sum()) / float(R.size + 1e-12)

            total_rec = float(R.sum()) + 1e-12
            ones_k = np.ones(det_len, dtype=np.int8)
            det_count = 0.0
            for row in R:
                det_count += float(np.sum(np.convolve(row, ones_k, mode="valid") >= det_len))
            det = float(det_count / total_rec)

            rr_list.append(rr)
            det_list.append(det)

        if not rr_list:
            return {"recurrence_rate": np.nan, "determinism": np.nan}

        return {
            "recurrence_rate": float(np.nanmean(rr_list)),
            "determinism": float(np.nanmean(det_list)),
        }


    def features_recurrence(self) -> Dict[str, float]:
        t0 = time.time()
        x = self._get_data_matrix()

        rr_vals, det_vals = [], []
        for ch in range(x.shape[0]):
            feats = self._recurrence_features_windowed(
                x[ch],
                sfreq=float(self.sfreq),
                m=self.recurrence_m,
                tau_ms=self.recurrence_tau_ms,
                eps=self.recurrence_eps,
                det_len=self.recurrence_det_len,
                target_fs=self.recurrence_target_fs,
                win_sec=self.recurrence_win_sec,
                overlap=self.recurrence_overlap,
                max_points=self.recurrence_max_points,
                max_seconds=self.recurrence_max_seconds,
            )
            rr_vals.append(feats["recurrence_rate"])
            det_vals.append(feats["determinism"])

        out: Dict[str, float] = {}
        out.update(self._summarize_vector(np.array(rr_vals, float), "recurrence", "rate"))
        out.update(self._summarize_vector(np.array(det_vals, float), "recurrence", "determinism"))
        self._qc_record("recurrence", t0, [np.array(rr_vals, float), np.array(det_vals, float)])
        return out

    

    def features_connectivity_entropy(self, band: str = "alpha") -> Dict[str, float]:
        key = f"connectivity.entropy_{band}"
        try:
            fc = self._compute_fc(band=band, method="wpli")
            iu = np.triu_indices_from(fc, k=1)
            vals = np.abs(fc[iu])
            if vals.size < 2:
                return {key: np.nan}
            s = float(vals.sum())
            if s <= 0 or not np.isfinite(s):
                return {key: np.nan}
            p = vals / s
            H = float(-np.sum(p * np.log(p + 1e-12)))
            Hmax = float(np.log(vals.size))
            return {key: (H / Hmax) if Hmax > 0 else np.nan}
        except Exception:
            return {key: np.nan}


    def features_temporal_dynamics(self, win_sec: float = 4.0, overlap: float = 0.5) -> Dict[str, float]:
        """
        Temporal variability of relative alpha band power and its lag-1 autocorrelation.
        """
        from scipy.stats import pearsonr
        data = self._get_data_matrix()
        sf = self.sfreq
        nwin = int(max(1, round(win_sec * sf)))
        step = max(1, int(round(nwin * (1.0 - overlap))))
        bd = self.bands

        rel_alpha_series = []
        for ch in range(data.shape[0]):
            vals = []
            for start in range(0, data.shape[1] - nwin + 1, step):
                seg = data[ch, start:start + nwin]
                f, P = welch(seg, fs=sf, nperseg=min(nwin, len(seg)), noverlap=0, scaling="density")
                m = _band_mask(f, bd.alpha)
                total = float(np.trapz(P, f)) + 1e-12
                vals.append(float(np.trapz(P[m], f[m]) / total) if np.any(m) else np.nan)
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) > 1:
                rel_alpha_series.append(vals)

        if not rel_alpha_series:
            return {"temporal.alpha_power_variance": np.nan,
                    "temporal.alpha_power_autocorr": np.nan}

        var_alpha = float(np.nanmean([np.var(v) for v in rel_alpha_series]))
        acorr_list = []
        for v in rel_alpha_series:
            if len(v) > 2:
                r, _ = pearsonr(v[:-1], v[1:])
                acorr_list.append(r)
        acorr_alpha = float(np.nanmean(acorr_list)) if acorr_list else np.nan

        return {
            "temporal.alpha_power_variance": var_alpha,
            "temporal.alpha_power_autocorr": acorr_alpha,
        }


    def features_fractal_spectrum(self) -> Dict[str, float]:
        freqs, psd = _psd_welch(self._get_data_matrix(), self.sfreq)
        logf = np.log10(freqs + 1e-12)
        curv = np.full(psd.shape[0], np.nan, float)
        for ch in range(psd.shape[0]):
            logp = np.log10(psd[ch] + 1e-12)
            if np.isfinite(logp).sum() >= 5:
                coeffs = np.polyfit(logf, logp, 2)
                curv[ch] = float(coeffs[0])
        return self._summarize_vector(curv, "fractal", "curvature")



    # ---------- Orchestrator ----------
    def extract_all(self) -> Dict[str, Any]:
        """Run all families; return dict with features, qc, config, and optional matrices."""
        features: Dict[str, float] = {}
        matrices: Dict[str, np.ndarray] = {}

        features.update(self.features_spectral())
        try:
            features.update(self.features_connectivity_graph(band="alpha"))
            if self.return_matrices:
                matrices["wpli_alpha"] = self._compute_fc("alpha", "wpli")
                matrices["pli_alpha"]  = self._compute_fc("alpha", "pli")
        except Exception as e:
            logger.warning(f"connectivity_graph failed: {e}")
        try:
            features.update(self.features_microstates(n_classes=4))
        except Exception as e:
            logger.warning(f"microstates failed: {e}")
        features.update(self.features_complexity())
        features.update(self.features_topography_asymmetry())
        try:
            features.update(self.features_cfc())
        except Exception as e:
            logger.warning(f"cfc failed: {e}")

        try:
            features.update(self.features_amplitude_envelope_connectivity())
        except Exception as e:
            logger.warning(f"aec failed: {e}")

        try:
            features.update(self.features_temporal_dynamics())
        except Exception as e:
            logger.warning(f"td failed: {e}")

        try:
            features.update(self.features_recurrence())
        except Exception as e:
            logger.warning(f"recurrence failed: {e}")

        try:
            features.update(self.features_connectivity_entropy())
        except Exception as e:
            logger.warning(f"ce failed: {e}")

        try:
            features.update(self.features_fractal_spectrum())
        except Exception as e:
            logger.warning(f"fs failed: {e}")

        payload = {
            "features": features,
            "qc": self._qc if self.qc_enabled else {},
            "config": {
                "per_channel": self.per_channel,
                "return_averages": self.return_averages,
                "return_matrices": self.return_matrices,
                "bands": {
                    "delta": self.bands.delta, "theta": self.bands.theta,
                    "alpha": self.bands.alpha, "beta": self.bands.beta, "gamma": self.bands.gamma,
                },
                "epoch_len": self.epoch_len,
                "epoch_overlap": self.epoch_overlap,
                "density": self.density,
                "random_state": self.random_state,
                "complexity_target_fs": self.complexity_target_fs,
                "complexity_max_seconds": self.complexity_max_seconds,
                "n_channels": len(self.ch_names),
                "sfreq": self.sfreq,
                                "recurrence_target_fs": self.recurrence_target_fs,
                "recurrence_win_sec": self.recurrence_win_sec,
                "recurrence_overlap": self.recurrence_overlap,
                "recurrence_m": self.recurrence_m,
                "recurrence_tau_ms": self.recurrence_tau_ms,
                "recurrence_eps": self.recurrence_eps,
                "recurrence_det_len": self.recurrence_det_len,
                "recurrence_max_points": self.recurrence_max_points,
                "recurrence_max_seconds": self.recurrence_max_seconds,

                
            },
        }

        if self.return_matrices:
            mats = {}
            try:
                mats["wpli_alpha"] = self._compute_fc("alpha", "wpli")
                mats["pli_alpha"]  = self._compute_fc("alpha", "pli")
            except Exception:
                pass
            if hasattr(self, "_last_conn"):
                mats["last_connectivity"] = self._last_conn
            if hasattr(self, "_last_cfc"):
                mats["last_cfc"] = self._last_cfc

            # serialize matrices safely
            from pathlib import Path
            payload["matrices"] = _serialize_matrix_dict(mats, Path("feature_matrices"))


        return payload


