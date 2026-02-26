## Simulating Jets at the CMS Detector using MadGraph + Pythia + Delphes

#### Authors: Aritra Bal, ETP
#### Contact: aritra.bal@do-not-spam-kit.edu

---

This pipeline generates simulated jet events by chaining three components: hard scattering
(MadGraph), parton shower and hadronization (Pythia8), and fast detector simulation (Delphes).

### Prerequisites

Working installations of the following are required:

- [ROOT](https://root.cern/)
- [LHAPDF](https://lhapdf.hepforge.org/)
- [MadGraph5\_aMC@NLO](https://launchpad.net/mg5amcnlo)
- [Pythia8](https://pythia.org/)
- [Delphes](https://cp3.irmp.ucl.ac.be/projects/delphes)
- [FastJet](http://fastjet.fr/)


The Docker container [neutrinoman4/hep-stack:latest](https://hub.docker.com/repository/docker/neutrinoman4/hep-stack/tags/latest/sha256-65ea12354bc93cf753736bc5e1d5ee457756aae7ba17aee648756617342b7033) (as of now, minimally tested) contains these packages and may be used. 

---

### 1. Hard Scattering with MadGraph (TTBar only)

MadGraph is only used for the $t\bar{t}$ sample. For QCD dijets, the hard scattering is handled
internally by Pythia8, and no MadGraph cards are provided for that process. For ease of usage, define an alias as: 

```bash
alias madgraph='/path/to/MG5_aMC_v3_7_0/bin/mg5_aMC'
```

Navigate to `MadGraph/ttbar/` and edit the output directory in `gen.dat` as needed, then run:

```bash
madgraph gen.dat
```

If MadSpin is needed to handle top quark decays, copy the decay card to the output directory:

```bash
cp madspin_card.dat <output_dir>/Cards/
```

Finally, run the launch command:

```bash
madgraph launch.dat
```

This produces an LHEF output that is passed to Pythia8 in the next step.

---

### 2. Parton Shower, Hadronization, and Detector Simulation

Showering, hadronization, and fast detector simulation are performed in a single step using the
`DelphesPythia8` executable bundled with Delphes. This requires Delphes to be compiled with
`HAS_PYTHIA8=true`. See the [Delphes + Pythia8 workbook](https://delphes.github.io/workbook/pythia8/)
for compilation and setup instructions.

The executable is run as:

```bash
/path/to/DelphesPythia8 \
    /path/to/DELPHES_CARD.tcl \
    /path/to/PYTHIA_CARD.cmnd \
    /path/to/output_tree.root
```

A Python driver script `PyDelphes/gen_reco.py` is provided to handle this step more conveniently.
Pythia8 configurations for QCD and TTBar are stored as JSON files in `PyDelphes/configs/`. The QCD
dijet configuration uses `frameType: 1` and runs standalone without any LHEF input from MadGraph.
The TTBar configuration expects the LHEF output from the MadGraph step above.

This simulation uses the CMS CP5 underlying event tune, along with the
`NNPDF3.1 nnlo_as_0118_luxqed` PDF set (LHAPDF ID `325100`). These are defined in
`PyDelphes/configs/defaults.py` and can be freely modified. The arguments in the config JSON
files are mostly self-explanatory.

---

### 3. Launching Jobs in Parallel

Once the configuration is in place, jobs can be launched in parallel on a local machine using
the script `PyDelphes/launcher.sh`:

```bash
source launcher.sh --config ./configs/config_QCD.json --jobs J --cores N
```

This launches `J` independent jobs using at most `N` cores. **Note:** I am not responsible for
any crashes arising from the use of this script. For large-scale production, always use a proper
batch system.

Seeds are set randomly via `np.random.randint`. To use a time-based seed instead, remove the
`--randomize` flag from the `xargs` launch command inside `launcher.sh`. Be careful to avoid
generating repetitive deterministic event samples when doing so.

---

### 4. nTuple Creation

Each job produces a single ROOT file `delphes.root` at a path determined by the config JSON.
This Delphes output tree must first be flattened into a standard ntuple.

The macro `PyDelphes/ntuplizer/makeNtuples.C` is adapted from the
[JetClass](https://github.com/jet-universe/particle_transformer) dataset ntuplizer. AI-generated
documentation is available in `docs.txt`, it is recommended to read this before running. Launch
the ntuplizer as:

```bash
root -l -q 'makeNtuples.C+("/path/to/delphes.root", "/path/to/flat_ntuple.root", "FatJet")'
```

The output is a per-jet ROOT ntuple `flat_ntuple.root` containing kinematic and substructure
variables.

---

### 5. Conversion to HDF5

The flat ntuple is converted to an HDF5 file using `h5_maker/ntuples_to_h5.py`:

```bash
python3 ntuples_to_h5.py --root-files /list/of/files.txt --max-cores N
```

The `--root-files` argument accepts a single path, multiple space-separated paths, or a `.txt`
file with one path per line. Files are processed in parallel using up to `N` cores.

Within this script, pairwise $\Delta R$ distances between jet constituents are computed alongside
$p_T$-weight contributions. These are required for computing Energy-Energy Correlators (EECs) at
arbitrary order in downstream analysis. The $\Delta R$ computation is the most **time-intensive**
step, and might be removed at a later stage since this is, in principle, easily derived on the fly. The final output is an HDF5 dataset containing all kinematic and derived quantities as
NumPy arrays.