"""
================================================================================
Pythia8 + Delphes Pipeline Controller
================================================================================
Accepts a user-supplied JSON config, merges it with DEFAULT_CONFIG, generates
a Pythia8 .cmnd file, and calls the DelphesPythia8 executable to produce a
detector-simulated ROOT output file.

Usage:
    python run_delphes.py --config /path/to/config.json [--no-verbose]

Author: Aritra Bal (ETP)
Date: 2026-02-17
================================================================================
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Import default configuration
from configs.defaults import DEFAULT_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Pythia8 + Delphes detector simulation pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to user config JSON file.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Whether to randomize the seed used for generation (will override any seed value in the config)."
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Suppress Pythia8 banner and initialisation output (default: verbose on).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config loading and merging
# ─────────────────────────────────────────────────────────────────────────────

def load_config(json_path: str) -> dict:
    """
    Load user JSON config and merge with DEFAULT_CONFIG.
    Merges each nested sub-dict independently so partial overrides work
    correctly — only the keys present in the user JSON are overridden.

    Parameters
    ----------
    json_path : str
        Path to user-supplied JSON config file.

    Returns
    -------
    config : dict
        Fully resolved configuration dict.
    """
    with open(json_path) as f:
        user_config = json.load(f)

    # Warn on unrecognised top-level keys
    for key in user_config:
        if key not in DEFAULT_CONFIG and key != "run_id":
            logger.warning(f"Unrecognised config key '{key}' — ignoring.")

    # Deep merge: per-group override
    config = {}
    for group, defaults in DEFAULT_CONFIG.items():
        if isinstance(defaults, dict):
            config[group] = {**defaults, **user_config.get(group, {})}
        else:
            config[group] = user_config.get(group, defaults)

    # run_id handled separately (top-level key, not nested)
    if "run_id" in user_config:
        config["run_id"] = user_config["run_id"]
    else:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config["run_id"] = run_id
        logger.info(f"No run_id specified — assigned automatically: {run_id}")

    return config

# ─────────────────────────────────────────────────────────────────────────────
# LHE file setup
# ─────────────────────────────────────────────────────────────────────────────

def prepare_lhe(config: dict, output_dir: Path) -> Path:
    """
    Decompress a gzipped LHE file to <output_dir>/tmp/unweighted_events.lhe.
    If the file is not gzipped, returns the original path unchanged.

    Parameters
    ----------
    config     : dict
    output_dir : Path

    Returns
    -------
    lhe_path : Path
        Path to the plain (decompressed) LHE file.
    """
    import gzip
    source = Path(config["paths"]["lhe_file"])
    if not source.suffix == ".gz":
        return source
    if '/' in config['run_id']:
        run_number = config['run_id'].split('/')[-1]
    else:
        run_number = config['run_id']
    # create a random temporary directory for the decompressed LHE file
    tmp_dir = Path(f'/tmp/abal/lhe_decompression/{run_number}')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    lhe_path = tmp_dir / f'unweighted_events_{run_number}.lhe'

    logger.info(f"Decompressing LHE file: {source} → {lhe_path}")
    with gzip.open(source, "rb") as f_in, open(lhe_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info("Decompression complete.")

    return lhe_path

# ─────────────────────────────────────────────────────────────────────────────
# Directory setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_directories(config: dict) -> tuple[Path, Path, Path]:
    """
    Resolve final output_dir and model_dir by appending run_id,
    create all required directories, and return their paths.

    Parameters
    ----------
    config : dict
        Fully merged configuration dict.

    Returns
    -------
    output_dir : Path
    model_dir  : Path
    cards_dir  : Path
    """
    run_id     = config["run_id"]
    output_dir = Path(config["paths"]["output_dir"]) / run_id
    model_dir  = Path(config["paths"]["model_dir"])  / run_id
    cards_dir  = model_dir / "cards"

    for d in (output_dir, model_dir, cards_dir):
        d.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ready: {d}")

    return output_dir, model_dir, cards_dir


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(model_dir: Path) -> None:
    """
    Configure loguru: stderr sink (already default) + file sink in model_dir.

    Parameters
    ----------
    model_dir : Path
        Directory where the log file will be written.
    """
    log_path = model_dir / "pipeline.log"
    logger.add(
        str(log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB",
    )
    logger.info(f"Log file: {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pythia8 .cmnd file generation
# ─────────────────────────────────────────────────────────────────────────────

def build_cmnd(config: dict, verbose: bool) -> str:
    """
    Build the Pythia8 .cmnd file content as a string from the merged config.

    Parameters
    ----------
    config : dict
        Fully merged configuration dict.
    verbose : bool
        If False, suppress Pythia8 banner and per-event output.

    Returns
    -------
    cmnd : str
        Full .cmnd file content.
    """
    r   = config["run"]
    t   = config["tune"]
    a   = config["alphas"]
    s   = config["shower"]
    m   = config["mpi"]
    h   = config["hadronization"]
    mis = config["misc"]
    p   = config["paths"]
    if r['lhe_frame_type'] == 4:
        decompressed_lhe_path = prepare_lhe(config, Path(p["output_dir"]))
        lines = [
        "! ── Pythia8 command file — generated by run_delphes.py ──────────────",
        "",
        "! Beam configuration",
        f"Beams:frameType       = {r['lhe_frame_type']}",
        f"Beams:LHEF            = {decompressed_lhe_path}",
        f"Beams:eCM             = {2 * r['ebeam']:.1f}",
        "",
        "! CP5 tune and PDF",
        f"Tune:pp               = {t['tune_pp']}",
        f"Tune:ee               = {t['tune_ee']}",
        f"PDF:pSet              = {t['pdf_pset']}",
        "",
        "! Alpha_s",
        f"SpaceShower:alphaSvalue  = {a['isr_value']}",
        f"SpaceShower:alphaSorder  = {a['isr_order']}",
        f"TimeShower:alphaSvalue   = {a['fsr_value']}",
        f"TimeShower:alphaSorder   = {a['fsr_order']}",
        f"MultipartonInteractions:alphaSvalue = {a['mpi_value']}",
        f"MultipartonInteractions:alphaSorder = {a['mpi_order']}",
        f"SigmaProcess:alphaSvalue = {a['process_value']}",
        f"SigmaProcess:alphaSorder = {a['process_order']}",
        "",
        "! Parton shower",
        f"PartonLevel:ISR       = {'on' if s['isr'] else 'off'}",
        f"PartonLevel:FSR       = {'on' if s['fsr'] else 'off'}",
        f"TimeShower:pTmin      = {s['ptmin_fsr']}",
        "",
        "! MPI and hadronization",
        f"PartonLevel:MPI       = {'on' if m['enabled'] else 'off'}",
        f"HadronLevel:all       = {'on' if h['enabled'] else 'off'}",
        "",
        "! Misc",
        f"SigmaTotal:zeroAXB    = {'on' if mis['sigma_total_zero_axb'] else 'off'}",
        "",
        "! Random seed",
        "Random:setSeed        = on",
        f"Random:seed           = {r['seed']}",
        "",
        "! Event count",
        f"Main:numberOfEvents   = {r['nevents']}",
        "Main:timesAllowErrors = 50",
        "",
    ]
    else: 
        logger.info("No LHE frame type 4 — skipping LHE preparation step.")
        lines = [
        "! ── Pythia8 command file — generated by run_delphes.py ──────────────",
        "",
        "! Beam configuration",
        f"Beams:frameType       = {r['lhe_frame_type']}",
        "Beams:idA             = 2212",
        "Beams:idB             = 2212",
        f"Beams:eCM             = {2 * r['ebeam']:.1f}",
        "",
        "! Hard QCD process",
        "HardQCD:all           = on",
        f"PhaseSpace:pTHatMin   = {r['pthat_min']}",
        "",
        "! CP5 tune and PDF",
        f"Tune:pp               = {t['tune_pp']}",
        f"Tune:ee               = {t['tune_ee']}",
        f"PDF:pSet              = {t['pdf_pset']}",
        "",
        "! Alpha_s",
        f"SpaceShower:alphaSvalue  = {a['isr_value']}",
        f"SpaceShower:alphaSorder  = {a['isr_order']}",
        f"TimeShower:alphaSvalue   = {a['fsr_value']}",
        f"TimeShower:alphaSorder   = {a['fsr_order']}",
        f"MultipartonInteractions:alphaSvalue = {a['mpi_value']}",
        f"MultipartonInteractions:alphaSorder = {a['mpi_order']}",
        f"SigmaProcess:alphaSvalue = {a['process_value']}",
        f"SigmaProcess:alphaSorder = {a['process_order']}",
        "",
        "! Parton shower",
        f"PartonLevel:ISR       = {'on' if s['isr'] else 'off'}",
        f"PartonLevel:FSR       = {'on' if s['fsr'] else 'off'}",
        f"TimeShower:pTmin      = {s['ptmin_fsr']}",
        "",
        "! MPI and hadronization",
        f"PartonLevel:MPI       = {'on' if m['enabled'] else 'off'}",
        f"HadronLevel:all       = {'on' if h['enabled'] else 'off'}",
        "",
        "! Misc",
        f"SigmaTotal:zeroAXB    = {'on' if mis['sigma_total_zero_axb'] else 'off'}",
        "",
        "! Random seed",
        "Random:setSeed        = on",
        f"Random:seed           = {r['seed']}",
        "",
        "! Event count",
        f"Main:numberOfEvents   = {r['nevents']}",
        "Main:timesAllowErrors = 50",
        "",
        ]
    

    if not verbose:
        lines += [
            "! Suppress output",
            "Print:quiet                    = on",
            "Init:showProcesses             = off",
            "Init:showMultipartonInteractions = off",
            "Init:showChangedSettings       = off",
            "Init:showChangedParticleData   = off",
            "Next:numberShowInfo            = 0",
            "Next:numberShowProcess         = 0",
            "Next:numberShowEvent           = 0",
            "",
        ]

    return "\n".join(lines)


def write_cmnd(config: dict, cards_dir: Path, verbose: bool) -> Path:
    """
    Write the .cmnd file to cards_dir and return its path.

    Parameters
    ----------
    config    : dict
    cards_dir : Path
    verbose   : bool

    Returns
    -------
    cmnd_path : Path
    """
    cmnd_content = build_cmnd(config, verbose)
    cmnd_path    = cards_dir / "pythia8.cmnd"
    cmnd_path.write_text(cmnd_content)
    logger.info(f"Pythia8 .cmnd file written: {cmnd_path}")
    return cmnd_path


# ─────────────────────────────────────────────────────────────────────────────
# Delphes card
# ─────────────────────────────────────────────────────────────────────────────

def stage_delphes_card(config: dict, cards_dir: Path) -> Path:
    """
    Copy the Delphes detector card to cards_dir for provenance and return
    the path to the local copy that will be passed to DelphesPythia8.

    Parameters
    ----------
    config    : dict
    cards_dir : Path

    Returns
    -------
    local_card : Path
    """
    source = Path(config["paths"]["delphes_card"])
    if not source.is_file():
        logger.error(f"Delphes card not found: {source}")
        sys.exit(1)

    local_card = cards_dir / source.name
    shutil.copy2(source, local_card)
    logger.info(f"Delphes card staged: {source} → {local_card}")
    return local_card


# ─────────────────────────────────────────────────────────────────────────────
# Settings summary
# ─────────────────────────────────────────────────────────────────────────────

def log_settings_summary(config: dict) -> None:
    """Log a human-readable summary of key Pythia8 and run settings."""
    r, t, a, s = config["run"], config["tune"], config["alphas"], config["shower"]
    logger.info("─── Pythia8 settings summary ────────────────────────────────")
    logger.info(f"  Run ID        : {config['run_id']}")
    logger.info(f"  LHE file      : {config['paths']['lhe_file']}")
    logger.info(f"  sqrt(s)       : {2 * r['ebeam']:.1f} GeV  (ebeam = {r['ebeam']} GeV)")
    logger.info(f"  N events      : {r['nevents']}")
    logger.info(f"  Seed          : {r['seed']}")
    logger.info(f"  Tune:pp / ee  : {t['tune_pp']} / {t['tune_ee']}  (CP5 base: Monash 2013)")
    logger.info(f"  PDF:pSet      : {t['pdf_pset']}  (NNPDF31_nnlo_as_0118_luxqed)")
    logger.info(f"  ISR / FSR     : {'on' if s['isr'] else 'off'} / {'on' if s['fsr'] else 'off'}")
    logger.info(f"  MPI           : {'on' if config['mpi']['enabled'] else 'off'}")
    logger.info(f"  Hadronization : {'on' if config['hadronization']['enabled'] else 'off'}")
    logger.info(f"  alpha_s (FSR) : {a['fsr_value']} (order {a['fsr_order']})")
    logger.info(f"  alpha_s (ISR) : {a['isr_value']} (order {a['isr_order']})")
    logger.info(f"  TimeShower pTmin : {s['ptmin_fsr']} GeV")
    logger.info(f"  Delphes card  : {config['paths']['delphes_card']}")
    logger.info("─────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# DelphesPythia8 execution
# ─────────────────────────────────────────────────────────────────────────────

def run_delphes(
    config:     dict,
    cmnd_path:  Path,
    card_path:  Path,
    output_dir: Path,
) -> None:
    """
    Call the DelphesPythia8 executable as a subprocess.

    Command signature:
        DelphesPythia8 <delphes_card> <output.root> <pythia8.cmnd>

    Parameters
    ----------
    config     : dict
    cmnd_path  : Path
    card_path  : Path
    output_dir : Path
    """
    executable  = config["paths"]["delphes_executable"]
    output_root = output_dir / "delphes.root"
    if output_root.exists():
        os.remove(output_root)
        logger.info(f"Removed the existing ROOT file at {output_root}")
    cmd = [executable, str(card_path), str(cmnd_path), str(output_root)]
    logger.info(f"Executing: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"DelphesPythia8 failed with return code {result.returncode}")
        sys.exit(1)
    
    logger.info(f"ROOT output written: {output_root}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main pipeline entry point."""
    args = parse_args()

    # ── Load and merge config ─────────────────────────────────────────────────
    config = load_config(args.config)
    # ── Directory setup ───────────────────────────────────────────────────────
    output_dir, model_dir, cards_dir = setup_directories(config)

    # ── Logging (file sink requires model_dir to exist first) ─────────────────
    setup_logging(model_dir)
    
    if args.randomize:
        import numpy as np
        rnd_seed = np.random.randint(1, 1e6)
        logger.info(f"Randomizing seed: {rnd_seed} (overriding any seed value in the config)")
        config["run"]["seed"] = rnd_seed
    
    # ── Settings summary ──────────────────────────────────────────────────────
    log_settings_summary(config)

    # ── Write .cmnd file ──────────────────────────────────────────────────────
    verbose   = not args.no_verbose
    cmnd_path = write_cmnd(config, cards_dir, verbose)

    # ── Stage Delphes card ────────────────────────────────────────────────────
    card_path = stage_delphes_card(config, cards_dir)

    # ── Save fully resolved config as provenance sidecar ─────────────────────
    provenance_path = model_dir / "config_resolved.json"
    with open(provenance_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Resolved config saved: {provenance_path}")

    # ── Run DelphesPythia8 ────────────────────────────────────────────────────
    run_delphes(config, cmnd_path, card_path, output_dir)
    tmp_dir = output_dir / "tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        logger.info(f"Removed temporary directory: {tmp_dir}")

if __name__ == "__main__":
    main()