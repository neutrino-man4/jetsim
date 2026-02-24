"""
================================================================================
Default configuration for the Pythia8 + Delphes pipeline.
================================================================================
Provides DEFAULT_CONFIG: a nested dict of all configurable parameters for the
LHE --> Pythia8 (CP5 tune, Run 3 settings) --> DelphesPythia8 pipeline.

Merge pattern for user overrides:
    config = {**DEFAULT_CONFIG, **user_supplied_dict}

Pythia8 CP5 settings from arXiv:1903.12179 recommendations.

Author: Aritra Bal (ETP)
Date: 2026-02-17
================================================================================
"""

DEFAULT_CONFIG: dict = {
    "run_id": "default_run",
    # ── Paths ─────────────────────────────────────────────────────────────────
    "paths": {
        "lhe_file":           "PLACEHOLDER_LHE_FILE",    # e.g. /ceph/abal/QFIT/MC/.../unweighted_events.lhe.gz
        "output_dir":        "PLACEHOLDER_OUTPUT_DIR for root files", # e.g. /ceph/abal/QFIT/MC/ppJJ_delphes.root
        "delphes_card":       "/work/abal/QFIT/DELPHES/Delphes-3.5.1/cards/delphes_card_CMS.tcl",
        "delphes_executable": "/work/abal/QFIT/DELPHES/Delphes-3.5.1/DelphesPythia8",           # assumed to be on PATH; supply full path if not
        "tmp_dir":            "/tmp/abal/",
        "model_dir": "PLACEHOLDER path for Pythia and Delphes cards, and also user logs for each run"
    },

    # ── Beam / run settings ───────────────────────────────────────────────────
    "run": {
        "ebeam":    6800.0,   # [GeV] per beam — LHC Run 3, sqrt(s) = 13.6 TeV
        "nevents":  500000,
        "seed":     0,       # 0 for random seed based on clock; set to fixed int for reproducibility
        "lhe_frame_type": 4,  # Pythia8 Beams:frameType — read kinematics from LHEF
    },

    # ── CP5 tune ──────────────────────────────────────────────────────────────
    # Base tune: Monash 2013 (Tune:pp=14), with CP5 parameter overrides.
    # PDF:pSet=20 --> NNPDF31_nnlo_as_0118_luxqed (Pythia8 internal numbering).
    # Ref: CMS-GEN-17-001, arXiv:1903.12179
    "tune": {
        "tune_pp":   14,      # Monash 2013 as base
        "tune_ee":   7,
        "pdf_pset":  20,      # NNPDF31_nnlo_as_0118_luxqed
    },

    # ── Alpha_s settings (CP5 values) ─────────────────────────────────────────
    "alphas": {
        "fsr_value":     0.118,
        "fsr_order":     2,
        "isr_value":     0.118,
        "isr_order":     2,
        "mpi_value":     0.118,
        "mpi_order":     2,
        "process_value": 0.118,
        "process_order": 2,
    },

    # ── Parton shower ─────────────────────────────────────────────────────────
    "shower": {
        "isr":         True,
        "fsr":         True,
        "ptmin_fsr":   0.5,   # TimeShower:pTmin [GeV] — Monash 2013 default
    },

    # ── Hadronization and MPI ─────────────────────────────────────────────────
    "hadronization": {
        "enabled": True,
    },
    "mpi": {
        "enabled": True,
    },

    # ── Misc CP5 flags ────────────────────────────────────────────────────────
    "misc": {
        "sigma_total_zero_axb": False,  # SigmaTotal:zeroAXB = off
    },

    # ── Jet settings (generator-level phase space cut in MadGraph run_card) ──
    # ptj_min is not a Pythia8 parameter — it is recorded here for provenance
    # and should be cross-checked against the run_card.dat ptj value.
    "jets": {
        "ptj_min":   20.0,    # [GeV] minimum parton pT in MadGraph run_card
    },
}


if __name__ == "__main__":
    import json
    # Pretty-print the full default config for inspection
    print(json.dumps(DEFAULT_CONFIG, indent=4))