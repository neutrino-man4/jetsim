"""
Read HDF5 files produced by the JetClass ROOT-to-HDF5 converter, recluster
jet constituents with the exclusive kT algorithm to find exactly 3 subjet
axes, and write augmented HDF5 files containing:

  - All original datasets (pass-through)
  - subjet_features      : (N, 3, 5)  ordered by subjet pT (descending)
                           features = [pT, eta, phi, pT_fraction, delta_r_to_jet]
  - subjet_feature_names : string array of length 5
  - jetFeatures          : (N, 13+3) = original 13 jet features plus
                           delta_r01, delta_r12, delta_r20

Author: Aritra Bal (ETP)
Date  : 2026-04-20
"""

import argparse
import logging
import os
import pathlib
import time
from multiprocessing import Pool
from typing import Optional

import fastjet
import h5py
import numpy as np
import vector
import awkward as ak

vector.register_awkward()
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_CONSTITUENTS: int = 100
N_SUBJETS: int = 3

# Radial thresholds for event-selection features (overridable via kwargs)
R_WIDE: float = 0.4   # wide-angle periphery boundary
R_COL: float  = 0.15  # collinear core boundary

SUBJET_FEATURE_NAMES: list[str] = [
    "subjet_pt",
    "subjet_eta",
    "subjet_phi",
    "subjet_pt_fraction",
    "subjet_delta_r_to_jet",
]

# Additional jet-level features appended after the original 13
EXTRA_JET_FEATURE_NAMES: list[str] = [
    "jet_delta_r01",
    "jet_delta_r12",
    "jet_delta_r20",
]

# The original 13 jet feature names (must match the order written by the
# converter script; we read them from the file but list them here for
# documentation purposes only).
_ORIGINAL_JET_FEATURE_NAMES: list[str] = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_energy",
    "jet_nparticles",
    "jet_sdmass",
    "jet_tau21",
    "jet_tau32",
    "jet_tau43",
    "jet_tau1",
    "jet_tau2",
    "jet_tau3",
    "jet_tau4",
]


# ---------------------------------------------------------------------------
# Delta-R helper
# ---------------------------------------------------------------------------
def delta_r(
    eta1: float | np.ndarray,
    phi1: float | np.ndarray,
    eta2: float | np.ndarray,
    phi2: float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute Delta_R = sqrt(deta^2 + dphi^2) with phi wrapping.

    Parameters
    ----------
    eta1, phi1, eta2, phi2 : scalar or ndarray
        Pseudorapidity and azimuthal angle [rad].

    Returns
    -------
    float or ndarray
        Delta_R value(s).
    """
    deta = eta1 - eta2
    dphi = phi1 - phi2
    # Wrap dphi to [-pi, pi]
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
    return np.sqrt(deta ** 2 + dphi ** 2)


# ---------------------------------------------------------------------------
# Exclusive kT subjet clustering
# ---------------------------------------------------------------------------

def cluster_subjets_kt_exclusive(
    part_deta: np.ndarray,
    part_dphi: np.ndarray,
    part_pt: np.ndarray,
    jet_pt: np.ndarray,
    n_subjets: int = 3,
) -> np.ndarray:
    """
    Vectorized exclusive-kT subjet reclustering over all events at once.

    Uses FastJet's awkward interface: all events are clustered in a single
    C++ call, eliminating per-event Python overhead. ~100-1000x faster than
    a per-event loop for N > 10^4.

    Coordinates are kept relative to the jet axis (delta-eta, delta-phi).
    The returned subjet eta/phi are therefore also relative to the jet axis.

    Parameters
    ----------
    part_deta : np.ndarray
        Shape (N, P). Constituent delta-eta relative to jet axis.
    part_dphi : np.ndarray
        Shape (N, P). Constituent delta-phi relative to jet axis.
    part_pt : np.ndarray
        Shape (N, P). Constituent pT [GeV]; padded entries are 0.
    jet_pt : np.ndarray
        Shape (N,). Parent jet pT [GeV], for pT-fraction computation.
    n_subjets : int
        Number of exclusive subjets per event.

    Returns
    -------
    np.ndarray
        Shape (N, n_subjets, 5), float32.
        Features: [pT, delta_eta, delta_phi, pT_fraction, delta_R_to_jet_axis].
        Ordered by descending subjet pT.
        Jets with too few real constituents: zero-padded rows.
    """
    N, P = part_pt.shape

    # Mask real constituents; drop padded entries
    real_mask = part_pt > 0.0

    # Build ragged awkward arrays of real constituents, one list per event
    # Each constituent as a Lorentz 4-vector in (pt, eta, phi, mass=0) form
    pt_ak  = ak.Array([part_pt[i][real_mask[i]]  for i in range(N)])
    eta_ak = ak.Array([part_deta[i][real_mask[i]] for i in range(N)])
    phi_ak = ak.Array([part_dphi[i][real_mask[i]] for i in range(N)])

    # Convert to px, py, pz, E (massless)
    px = pt_ak * np.cos(phi_ak)
    py = pt_ak * np.sin(phi_ak)
    pz = pt_ak * np.sinh(eta_ak)
    E  = pt_ak * np.cosh(eta_ak)

    constituents = ak.zip(
        {"px": px, "py": py, "pz": pz, "E": E},
        with_name="Momentum4D",
    )

    # Vectorized clustering: single C++ call handles all events
    jet_def = fastjet.JetDefinition(fastjet.kt_algorithm, 1.5)
    cs = fastjet.ClusterSequence(constituents, jet_def)

    # Exclusive mode: exactly n_subjets per event
    subjets = cs.exclusive_jets(n_jets=n_subjets)
    # subjets shape: (N, n_subjets) with fields px, py, pz, E

    # Extract kinematics; fastjet returns subjets sorted by pT descending by default
    sj_pt  = np.sqrt(subjets.px**2 + subjets.py**2)
    sj_eta = np.arcsinh(subjets.pz / sj_pt)
    sj_phi = np.arctan2(subjets.py, subjets.px)

    # To numpy with shape (N, n_subjets)
    sj_pt_np  = ak.to_numpy(sj_pt).astype(np.float32)
    sj_eta_np = ak.to_numpy(sj_eta).astype(np.float32)
    sj_phi_np = ak.to_numpy(sj_phi).astype(np.float32)

    # pT fraction relative to parent jet
    jet_pt_safe = np.where(jet_pt > 0.0, jet_pt, 1.0).astype(np.float32)
    sj_pt_frac = sj_pt_np / jet_pt_safe[:, np.newaxis]

    # delta_R to jet axis = origin in the relative frame = sqrt(eta^2 + phi^2)
    # Note: sj_phi is in [-pi, pi]; in the relative frame, jet axis sits at (0,0)
    # but constituents near phi = +/- pi could wrap. Guard with minimal-image wrap.
    sj_phi_wrapped = (sj_phi_np + np.pi) % (2 * np.pi) - np.pi
    dr_to_jet = np.sqrt(sj_eta_np**2 + sj_phi_wrapped**2).astype(np.float32)

    # Sanity: in rare cases fewer than n_subjets can be returned
    # (FastJet errors on events with < n_subjets constituents). Pre-filter or
    # catch that separately; in production, drop those events upstream.

    # Assemble (N, n_subjets, 5)
    result = np.stack(
        [sj_pt_np, sj_eta_np, sj_phi_wrapped, sj_pt_frac, dr_to_jet],
        axis=-1,
    ).astype(np.float32)

    return result


# ---------------------------------------------------------------------------
# Pairwise opening angles between subjets
# ---------------------------------------------------------------------------
def subjet_opening_angles(subjet_features: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Delta_R between subjet axes (pairs 01, 12, 20).

    Parameters
    ----------
    subjet_features : np.ndarray
        Shape (N, 3, F).  subjet_features[:, :, 1] = eta,
                          subjet_features[:, :, 2] = phi.

    Returns
    -------
    np.ndarray
        Shape (N, 3), float32.  Columns: [dR_01, dR_12, dR_20].
    """
    N = subjet_features.shape[0]
    eta = subjet_features[:, :, 1]  # (N, 3)
    phi = subjet_features[:, :, 2]  # (N, 3)

    dr01 = delta_r(eta[:, 0], phi[:, 0], eta[:, 1], phi[:, 1])
    dr12 = delta_r(eta[:, 1], phi[:, 1], eta[:, 2], phi[:, 2])
    dr20 = delta_r(eta[:, 2], phi[:, 2], eta[:, 0], phi[:, 0])

    return np.stack([dr01, dr12, dr20], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Event-selection helpers
# ---------------------------------------------------------------------------

def _nearest_axis_dr(
    part_pt: np.ndarray,
    part_deta: np.ndarray,
    part_dphi: np.ndarray,
    sj_eta: np.ndarray,
    sj_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized dR from every constituent to its nearest subjet axis.

    Parameters
    ----------
    part_pt   : (N, P) — constituent pT; zero for padded slots
    part_deta : (N, P) — constituent delta-eta relative to jet axis
    part_dphi : (N, P) — constituent delta-phi relative to jet axis
    sj_eta    : (N, K) — subjet eta (same relative frame)
    sj_phi    : (N, K) — subjet phi (same relative frame)

    Returns
    -------
    dR_min : (N, P) float32 — dR to nearest axis; padded slots set to inf
    real   : (N, P) bool   — True for non-padded constituents
    """
    real = part_pt > 0.0                              # (N, P)

    # Broadcast (N, P, 1) vs (N, 1, K) → (N, P, K)
    deta = part_deta[:, :, np.newaxis] - sj_eta[:, np.newaxis, :]
    dphi = part_dphi[:, :, np.newaxis] - sj_phi[:, np.newaxis, :]
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
    dR   = np.sqrt(deta ** 2 + dphi ** 2)             # (N, P, K)

    dR_min = dR.min(axis=2)                            # (N, P)
    dR_min = np.where(real, dR_min, np.inf)            # hide padded slots
    return dR_min.astype(np.float32), real


def compute_event_selections(
    part_pt: np.ndarray,
    part_deta: np.ndarray,
    part_dphi: np.ndarray,
    subjet_feats: np.ndarray,
    jet_features_aug: np.ndarray,
    q: float,
    r_wide: float = R_WIDE,
    r_col: float = R_COL,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute six boolean event-selection masks defining deformation directions
    δμ(k) = ⟨O⟩_{S_k} − ⟨O⟩_all.

    Parameters
    ----------
    part_pt          : (N, P) constituent pT [GeV]; zero-padded
    part_deta        : (N, P) constituent delta-eta relative to jet axis
    part_dphi        : (N, P) constituent delta-phi relative to jet axis
    subjet_feats     : (N, 3, 5) — [pT, eta, phi, pT_frac, dR_jet] per subjet
    jet_features_aug : (N, 16)  — augmented jet features; tau1..4 at cols 9..12
    q                : float in (0, 1) — selection fraction
    r_wide           : wide-angle radial threshold (default R_WIDE = 0.4)
    r_col            : collinear radial threshold  (default R_COL  = 0.15)

    Returns
    -------
    masks : (N, 6) bool — one column per selection; mask.sum(axis=0) ≈ q * N
    names : list[str] of length 6
    """
    sj_eta = subjet_feats[:, :, 1]   # (N, 3)
    sj_phi = subjet_feats[:, :, 2]   # (N, 3)

    dR_min, _ = _nearest_axis_dr(part_pt, part_deta, part_dphi, sj_eta, sj_phi)
    # Padded slots: pT=0 and dR_min=inf, so they contribute nothing to sums.

    # --- s1: wide-angle enhancement ------------------------------------------
    s1: np.ndarray = (part_pt * (dR_min > r_wide)).sum(axis=1)

    # --- s2: collinear hardening ---------------------------------------------
    s2: np.ndarray = (part_pt * (dR_min < r_col)).sum(axis=1)

    # --- s3: opening-angle shift (dR between the two leading subjet axes) ----
    # Subjets are sorted pT-descending; "leading two" → indices 0 and 1.
    s3: np.ndarray = delta_r(
        sj_eta[:, 0], sj_phi[:, 0],
        sj_eta[:, 1], sj_phi[:, 1],
    )

    # --- s4: prong asymmetry (variance of momentum fractions) ----------------
    sj_pt     = subjet_feats[:, :, 0]                   # (N, 3)
    sj_pt_sum = sj_pt.sum(axis=1, keepdims=True)        # (N, 1)
    sj_pt_sum = np.where(sj_pt_sum > 0.0, sj_pt_sum, 1.0)
    z         = sj_pt / sj_pt_sum                       # (N, 3) momentum fractions
    s4: np.ndarray = z.var(axis=1)                      # (N,)

    # --- s5: inter-prong bridge ----------------------------------------------
    mid_mask = (dR_min > r_col) & (dR_min < r_wide)    # (N, P)
    s5: np.ndarray = (part_pt * mid_mask).sum(axis=1)

    # --- s6: soft 4th-prong contamination (τ4/τ3) ---------------------------
    # tau3 and tau4 are at columns 11 and 12 in the augmented feature array.
    tau3 = jet_features_aug[:, 11].astype(np.float64)
    tau4 = jet_features_aug[:, 12].astype(np.float64)
    tau3_safe = np.where(tau3 > 0.0, tau3, np.finfo(np.float64).tiny)
    s6: np.ndarray = tau4 / tau3_safe

    # --- Quantile masks -------------------------------------------------------
    def _top_q(s: np.ndarray) -> np.ndarray:
        return s >= np.quantile(s, 1.0 - q)

    def _bot_q(s: np.ndarray) -> np.ndarray:
        return s <= np.quantile(s, q)

    masks = np.stack(
        [_top_q(s1), _top_q(s2), _top_q(s3), _top_q(s4), _top_q(s5), _bot_q(s6)],
        axis=1,
    )  # (N, 6) bool

    names: list[str] = [
        "wide_angle",
        "collinear_hardening",
        "opening_angle",
        "prong_asymmetry",
        "inter_prong_bridge",
        "soft_4th_prong",
    ]

    return masks.astype(bool), names


# ---------------------------------------------------------------------------
# Core per-file processing
# ---------------------------------------------------------------------------
def process_file(task: tuple) -> None:
    """
    Read one HDF5 file, compute subjets and augmented jet features, and write
    a new HDF5 file.

    Parameters
    ----------
    task : tuple
        (input_path, output_path)
    """
    input_path: str
    output_path: str
    q: float
    input_path, output_path, q = task

    logger.info("Processing %s --> %s", input_path, output_path)

    with h5py.File(input_path, "r") as fin:
        # ---- Read constituent arrays ----------------------------------------
        # jetConstituentsList : (N, 100, 3) = [deta, dphi, pt]
        jet_pfc = fin["jetConstituentsList"][:]          # float32

        part_deta = jet_pfc[:, :, 0]   # (N, 100)
        part_dphi = jet_pfc[:, :, 1]   # (N, 100)
        part_pt   = jet_pfc[:, :, 2]   # (N, 100)

        # ---- Read jet features ----------------------------------------------
        # jetFeatures : (N, 13) = [pt, eta, phi, E, npart, sdmass, tau21, tau32, tau43, tau1..4]
        jet_features = fin["jetFeatures"][:]             # float32 (N, 13)

        # Read jet-level scalars we need for absolute coordinates
        jet_eta = jet_features[:, 1]   # column index 1 = jet_eta
        jet_phi = jet_features[:, 2]   # column index 2 = jet_phi
        jet_pt  = jet_features[:, 0]   # column index 0 = jet_pt

        # Read original jet feature names to extend them
        try:
            orig_jet_feature_names: list[str] = [
                n.decode() if isinstance(n, bytes) else n
                for n in fin["jetFeatureNames"][:]
            ]
        except KeyError:
            orig_jet_feature_names = _ORIGINAL_JET_FEATURE_NAMES[:]

        # ---- Pass-through all other datasets --------------------------------
        passthrough: dict[str, np.ndarray] = {}
        skip_keys = {"jetFeatures", "jetFeatureNames"}
        for key in fin.keys():
            if key not in skip_keys:
                passthrough[key] = fin[key][:]

    N = part_pt.shape[0]
    logger.info("  %d jets loaded", N)

    # ---- Recluster to 3 subjets -------------------------------------------
    logger.info("  Clustering exclusive kT subjets (N=3) ...")
    subjet_feats = cluster_subjets_kt_exclusive(
        part_deta, part_dphi, part_pt, jet_pt,
        n_subjets=N_SUBJETS,
    )  # (N, 3, 5)

    # ---- Pairwise opening angles ------------------------------------------
    dr_pairs = subjet_opening_angles(subjet_feats)  # (N, 3): dR01, dR12, dR20

    # ---- Augmented jet features : (N, 13+3) ---------------------------------
    jet_features_aug = np.concatenate(
        [jet_features, dr_pairs], axis=-1
    ).astype(np.float32)  # (N, 16)

    extended_jet_feature_names: list[str] = (
        orig_jet_feature_names + EXTRA_JET_FEATURE_NAMES
    )

    # ---- Event-selection masks -------------------------------------------
    logger.info("  Computing event-selection masks (q=%.3f) ...", q)
    sel_masks, sel_names = compute_event_selections(
        part_pt, part_deta, part_dphi,
        subjet_feats, jet_features_aug,
        q=q,
    )  # (N, 6), list[str]

    # ---- Sanity checks -----------------------------------------------------
    assert not np.isnan(subjet_feats).any(),      "NaN in subjet_features"
    assert not np.isnan(jet_features_aug).any(),  "NaN in augmented jetFeatures"

    # ---- Write output HDF5 -------------------------------------------------
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    h5_str_dtype = h5py.special_dtype(vlen=str)

    with h5py.File(output_path, "w") as fout:
        # Pass-through datasets
        for key, data in passthrough.items():
            fout.create_dataset(key, data=data)

        # Augmented jet features
        fout.create_dataset("jetFeatures", data=jet_features_aug)
        fout.create_dataset(
            "jetFeatureNames",
            data=np.array(extended_jet_feature_names, dtype=h5_str_dtype),
        )

        # New subjet datasets
        fout.create_dataset("subjetFeatures", data=subjet_feats)
        fout.create_dataset(
            "subjetFeatureNames",
            data=np.array(SUBJET_FEATURE_NAMES, dtype=h5_str_dtype),
        )

        # Event-selection masks: (N, 6) bool
        fout.create_dataset("deformationMasks", data=sel_masks)
        fout.create_dataset(
            "deformationNames",
            data=np.array(sel_names, dtype=h5_str_dtype),
        )

    logger.info("  Written: %s  (%d jets)", output_path, N)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Recluster jet constituents to exclusive kT subjets and augment "
            "HDF5 files with subjet_features and extended jet features."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        metavar="PATH",
        help=(
            "Path to a single HDF5 file, OR a .txt file containing one HDF5 "
            "file path per line."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        metavar="DIR",
        help="Root output directory.  Run-ID subdirectories are created automatically.",
    )
    parser.add_argument(
        "--jet-type",
        type=str,
        default="TTBar",
        metavar="TYPE",
        help="Jet type tag used to name output files (default: ttbar).",
    )
    parser.add_argument(
        "--max-cores",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--selection-q",
        type=float,
        default=0.25,
        metavar="Q",
        help=(
            "Quantile fraction for event-selection masks, in (0, 1). "
            "Each selection picks the top (or bottom) Q fraction of a feature score. "
            "Default: 0.25 (top quartile)."
        ),
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help=(
            "Process only the first file and write to /tmp/abal/subjets_test.h5 "
            "without spawning workers."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# File-list resolution  (mirrors converter logic)
# ---------------------------------------------------------------------------
def resolve_file_list(raw_input: str) -> list[str]:
    """
    Resolve --input into a flat list of HDF5 file paths.

    If the argument ends in '.txt', it is treated as a manifest file with one
    path per line; otherwise it is treated as a direct file path.

    Parameters
    ----------
    raw_input : str
        Value of args.input.

    Returns
    -------
    list[str]
        Ordered, deduplicated list of HDF5 file paths.

    Raises
    ------
    FileNotFoundError
        If any resolved path does not exist on disk.
    """
    if raw_input.endswith(".txt"):
        with open(raw_input, "r") as fh:
            paths = [line.strip() for line in fh if line.strip()]
        logger.info("Read %d paths from %s", len(paths), raw_input)
    else:
        paths = [raw_input]

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"HDF5 files not found: {missing}")

    return list(dict.fromkeys(paths))  # preserve order, deduplicate


def build_output_path(
    input_path: str,
    output_root: str,
    jet_type: str, # unused for now, but could be used to further customize output file naming
    index: int,
) -> str:
    """
    Construct the output HDF5 path mirroring the converter's run_XX/jet_type.h5
    layout.

    The run ID is extracted as the name of the parent directory of the input
    file (e.g. .../ttbar/run_04/ttbar.h5 --> run_04).  If extraction fails, a
    zero-padded index is used as fallback.

    Parameters
    ----------
    input_path : str
        Full path to the input HDF5 file.
    output_root : str
        Root output directory supplied via --output.
    jet_type : str
        Jet-type tag used to name the output file [UNUSED currently] TODO: Remove
    index : int
        Sequential index used as fallback run ID.

    Returns
    -------
    str
        Full path to the output HDF5 file.
    """
    try:
        run_id = pathlib.Path(input_path).parent.name
    except Exception as exc:
        logger.warning("Could not extract run_id from '%s': %s", input_path, exc)
        run_id = f"run_{index:02d}"

    out_dir = pathlib.Path(output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"jet_data.h5")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Main entry point."""
    args = parse_args()

    file_list: list[str] = sorted(resolve_file_list(args.input))
    logger.info("Total HDF5 files to process: %d", len(file_list))

    q: float = args.selection_q
    if not (0.0 < q < 1.0):
        raise ValueError(f"--selection-q must be in (0, 1), got {q}")

    if args.test_run:
        out = "/tmp/abal/subjets_test.h5"
        pathlib.Path("/tmp/abal").mkdir(parents=True, exist_ok=True)
        logger.info("TEST RUN --> %s", out)
        process_file((file_list[0], out, q))
        return

    # Build task list
    tasks: list[tuple] = []
    for i, fpath in enumerate(file_list):
        out_path = build_output_path(fpath, args.output, args.jet_type, i)
        tasks.append((fpath, out_path, q))

    n_cores = min(args.max_cores, len(tasks))
    logger.info("Using %d worker(s) for %d file(s)", n_cores, len(tasks))

    if n_cores <= 1:
        for task in tasks:
            process_file(task)
    else:
        with Pool(processes=n_cores) as pool:
            pool.map(process_file, tasks)

    logger.info("All files processed.")


if __name__ == "__main__":
    main()