# clusterMD

**Automated density-based clustering and structural analysis of multidimensional PCA/dPCA projections derived from molecular-dynamics trajectories using HDBSCAN.**

`clusterMD` is a command-line workflow for clustering molecular-dynamics projection data, characterizing density-defined conformational clusters, extracting representative structures, and summarizing observed changes between assigned cluster labels along trajectory order.

The workflow is designed primarily for multidimensional PCA/dPCA projection data produced from molecular-dynamics trajectories, including projection files generated with `carma` or through `grcarma` workflows.

<p align="center">
  <img src="assets/clusterMD_graphical_abstract.png" alt="clusterMD graphical abstract showing trajectory data, PCA/dPCA projection, HDBSCAN clustering, representative conformations, a population-derived free-energy landscape, and observed label changes." width="100%">
</p>

<p align="center"><em>Graphical overview of the clusterMD workflow. HDBSCAN clustering is performed in the retained multidimensional PCA/dPCA space; the 2D free-energy landscape is shown for visualization. Arrows summarize observed changes between assigned cluster labels and should not be interpreted as kinetic rate estimates.</em></p>

---

## Overview

`clusterMD` integrates multidimensional clustering, trajectory-aware frame handling, structural extraction, structural fitting, descriptive free-energy visualization, and chronological label-change analysis into a single workflow.

The main stages are:

1. **Projection ingestion**
   Reads multidimensional PCA/dPCA coordinates and preserves their associated trajectory-frame identifiers.

2. **Density-based clustering**
   Applies HDBSCAN in the retained multidimensional projection space to identify density-defined conformational clusters while explicitly retaining unassigned/noise points.

3. **Cluster characterization**
   Calculates cluster populations, descriptive population-derived free-energy estimates, and multidimensional clustering summaries.

4. **Structural characterization**
   Selects one observed representative frame per cluster, extracts cluster-member structures from the original trajectory, performs structural fitting and averaging with `carma`, and optionally applies downstream RMSD-based structural filtering.

5. **Chronological label-change analysis**
   Summarizes observed changes between assigned cluster labels along trajectory order and generates descriptive tables and plots for inspection.

> **Important:** HDBSCAN clusters should not automatically be interpreted as metastable states, and observed changes between assigned cluster labels are not kinetic rate estimates.

---

## Key features

- Multidimensional PCA/dPCA clustering with HDBSCAN
- Support for both `eom` and `leaf` HDBSCAN cluster-selection methods
- Manual or automatic HDBSCAN parameter selection
- Preservation of original trajectory frame identifiers
- Explicit retention of HDBSCAN unassigned/noise points
- Cluster population statistics
- Population-derived free-energy estimates
- 2D projection visualizations
- Chronological observed-label-change summaries
- Representative observed-frame extraction
- `carma`-based structural fitting and averaging
- Cluster structural superpositions
- Optional RMSD-based downstream structural filtering
- Structural processing restricted by segment identifier using `--segid`
- Cluster-only execution mode
- Post-run observed-label-change inspection
- Runtime provenance and software-version reporting
- Deterministic input-file selection with warnings when multiple candidates are present

---

## Scientific interpretation

`clusterMD` identifies **density-defined conformational clusters** in PCA/dPCA projection space.

The software does not, by itself, establish that those clusters correspond to thermodynamic states or metastable states.

Likewise, chronological changes between assigned cluster labels are descriptive observations of trajectory ordering. They should not be interpreted directly as:

- kinetic transition rates,
- Markov transition probabilities,
- state lifetimes,
- free-energy barriers,
- or proof of folding/unfolding pathways.

Population-derived free-energy estimates are descriptive and depend on the assumptions and quality of the underlying molecular-dynamics sampling. Their interpretation is strongest when the sampled trajectory is representative of the relevant equilibrium ensemble.

Clustering is performed in the retained multidimensional PCA/dPCA coordinate space. Two-dimensional figures are used for visualization only and are not necessarily the full space in which HDBSCAN clustering was performed.

---

## Structural representative definition

For each cluster, `clusterMD`:

1. calculates the arithmetic centroid of the cluster in the retained PCA/dPCA coordinate space,
2. identifies the observed cluster frame nearest that centroid,
3. extracts that observed trajectory frame as the representative structure.

The reported representative is therefore:

**the observed frame nearest the arithmetic centroid in the retained PCA/dPCA coordinate space**

and should not be described as a mathematical medoid.

---

## External dependency: carma

Structural processing in `clusterMD` relies on **carma**, a stand-alone molecular-dynamics analysis program designed primarily for DCD/PSF trajectories.

`carma` provides functionality including:

- Cartesian PCA,
- dihedral-angle PCA,
- trajectory fitting,
- structural averaging,
- RMSD analysis,
- atomic fluctuation analysis,
- trajectory superposition,
- and other molecular-dynamics analysis operations.

`clusterMD` uses selected `carma` functionality for structural characterization of HDBSCAN-defined conformational clusters.

`grcarma` is the graphical user interface to `carma` and can be used to automate and facilitate upstream molecular-dynamics analyses, including PCA/dPCA workflows.

For structural analysis, the `carma` executable must be installed separately and available in the system `PATH`.

The clusterMD validation environment used:

```text
carma 2.3
````

`carma` and `grcarma` are external software and are not bundled with `clusterMD`.

---

## Python requirements

The current validated environment uses:

```
Python          3.12.3
NumPy           1.26.4
SciPy           1.17.1
scikit-learn    1.9.1
fast-hdbscan    0.3.2
Numba           0.68.0
NetworkX        3.7
Matplotlib      3.11.2
psutil          7.2.2
```

The exact validated Python environment is recorded in:

```
requirements-lock.txt
```

The broader installation requirements are listed in:

```
requirements.txt
```

The exact lock file is intended for reproducibility of the validated release environment.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── setup_env.sh
└── scripts/
    └── clusterMD
```

Future release packaging may additionally include structured documentation, test fixtures, citation metadata, examples, and release metadata.

---

## Installation

Clone the repository:

```
git clone https://github.com/tasosmelas/md-hdbscan-clustering.git
cd md-hdbscan-clustering
```

Create a virtual environment:

```
python3 -m venv .venv
```

Activate it.

For Bash/Zsh:

```
source .venv/bin/activate
```

For fish:

```
source .venv/bin/activate.fish
```

Install the supported Python dependencies:

```
python -m pip install -r requirements.txt
```

For exact reproduction of the validated Python environment:

```
python -m pip install -r requirements-lock.txt
```

Alternatively, the provided setup script can be used from the repository root:

```
chmod +x setup_env.sh
./setup_env.sh
```

The setup script:

- checks that the repository structure is present,
- makes scripts in `scripts/` executable,
- installs Python dependencies from `requirements.txt`,
- prints the path command required to make `clusterMD` directly executable from the shell.

To make `clusterMD` available from anywhere, add the repository `scripts/` directory to your `PATH`.

For Bash/Zsh:

```
export PATH="/path/to/md-hdbscan-clustering/scripts:$PATH"
```

For fish:

```
set -gx PATH /path/to/md-hdbscan-clustering/scripts $PATH
```

---

## Input data

The primary input is a whitespace-delimited PCA/dPCA projection file.

The first column contains trajectory-frame identifiers.

The remaining columns contain projected coordinates.

Typical filenames include:

```
carma.PCA.fluctuations.dat
carma.dPCA.fluctuations.dat
```

A simplified example:

```
1       -0.8342    1.4410    0.2184
4       -0.7915    1.3922    0.1947
7       -0.7551    1.3504    0.1821
10      -0.7327    1.3118    0.1654
```

The frame identifiers are preserved through clustering and are used to map cluster assignments back to the molecular-dynamics trajectory.

When structural analysis is requested, the corresponding trajectory and topology files must also be available.

Typical structural inputs are:

```
*.dcd
*.psf
```

---

## Projection stride and frame identifiers

`clusterMD` preserves the frame identifiers present in the projection file.

If the projection itself was generated with a stride upstream, for example with `carma -step 3`, then the projection file may contain frame identifiers such as:

```
1
4
7
10
...
```

These identifiers correspond to the original trajectory-frame indexing represented in the projection file.

The `--step` option in `clusterMD` applies an additional stride during projection-file ingestion.

This distinction is important:

- upstream `carma` stride changes which frames are written to the projection file,
- `clusterMD --step` subsamples the rows already present in that projection file.

---

## Basic usage

Run the full pipeline from a directory containing, or adjacent to, the projection file:

```
clusterMD
```

Example:

```
clusterMD --auto --method leaf --dim 5 --rmsd 2.0 --segid A
```

Other examples:

```
clusterMD --auto
```

```
clusterMD --method leaf
```

```
clusterMD --dim 5
```

```
clusterMD --step 5
```

```
clusterMD --rmsd 2.0
```

```
clusterMD --segid A
```

```
clusterMD --cluster-only
```

For the complete command-line interface:

```
clusterMD --help
```

---

## Command-line options

### `--size`

```
--size INT
```

Sets HDBSCAN `min_cluster_size`.

Default:

```
500
```

---

### `--samples`

```
--samples INT
```

Sets HDBSCAN `min_samples`.

Default:

```
50
```

---

### `--auto`

```
--auto
```

Automatically sets `min_cluster_size` and `min_samples` using the built-in frame-count heuristic.

---

### `--method`

```
--method {eom,leaf}
```

Selects the HDBSCAN cluster-selection method.

Supported values:

```
eom
leaf
```

Default:

```
eom
```

---

### `--dim`

```
--dim INT
```

Selects the number of projected dimensions used in clustering.

```
0
```

uses all available projected dimensions.

---

### `--step`

```
--step INT
```

Applies a row stride during projection-file ingestion.

Default:

```
1
```

---

### `--rmsd`

```
--rmsd FLOAT
```

Applies an optional RMSD threshold in Ångström during downstream structural filtering.

Default:

```
0
```

which disables RMSD filtering.

RMSD filtering does not alter the original HDBSCAN cluster assignments.

---

### `--temp`

```
--temp FLOAT
```

Temperature in Kelvin used for population-derived free-energy estimates.

Default:

```
298.0
```

---

### `--segid`

```
--segid STRING
```

Restricts structural extraction and fitting to the specified segment identifier.

Example:

```
clusterMD --segid A
```

The option affects structural processing only. It does not alter the HDBSCAN clustering performed on the projection coordinates.

---

### `--cluster-only`

```
--cluster-only
```

Runs projection ingestion and HDBSCAN clustering, exports the assignments, and exits before downstream visualization and structural analysis.

This is useful for clustering-focused workflows and validation.

---

### `--top`

```
--top INT
```

Controls how many of the most frequent observed label changes are displayed in diagnostic tables and plots.

Default:

```
15
```

A value of:

```
0
```

disables displayed edges.

---

### `--transitions`

```
--transitions
```

These are aliases.

With no cluster IDs, the option displays the most frequent observed changes.

Example:

```
clusterMD --transitions
```

A specific pair can be inspected by providing two cluster IDs:

```
clusterMD --transitions 0 5
```

---

### `--events`

```
--events INT
```

Maximum number of chronological label-change events printed during diagnostic inspection.

Default:

```
3
```

---

## Automatic HDBSCAN parameter heuristic

When `--auto` is enabled, `clusterMD` derives HDBSCAN parameters from the number of analyzed projection rows.

The current heuristic is:

```
min_cluster_size = max(100, int(500 * sqrt(N / 1,000,000)))
min_samples      = max(10, int(min_cluster_size / 10))
```

where `N` is the number of analyzed projection rows.

This heuristic is intended as a practical default.

It should not be interpreted as a universally optimal parameter-selection rule.

Parameter sensitivity should be examined when scientific conclusions depend strongly on cluster boundaries or cluster counts.

---

## HDBSCAN clustering output

The main cluster-assignment file follows the general naming pattern:

```
hdbscan.<D>-D.clusters.dat
```

where `<D>` is the number of retained dimensions.

The file contains:

1. original trajectory-frame identifier,
2. cluster label,
3. retained PCA/dPCA coordinates.

Unassigned/noise points retain the HDBSCAN label:

```
-1
```

`clusterMD` preserves this label rather than forcibly assigning noise points to clusters.

---

## Cluster populations

For each cluster, `clusterMD` calculates the number of assigned projection frames.

The total population satisfies:

```
assigned frames + unassigned/noise frames = analyzed frames
```

Cluster relabeling does not alter cluster membership.

---

## Free-energy estimates

Population-derived free-energy estimates are calculated from observed populations at the selected temperature.

These are descriptive estimates derived from sampling frequency.

Their interpretation depends on the sampling characteristics of the underlying molecular-dynamics trajectory.

They should not be treated as exact thermodynamic free energies when the trajectory does not adequately represent equilibrium sampling.

Free-energy visualizations are intended to support interpretation of sampled conformational density.

---

## Two-dimensional visualizations

`clusterMD` generates two-dimensional projection figures for inspection and presentation.

These figures do not imply that clustering was performed only in two dimensions.

HDBSCAN clustering occurs in the retained multidimensional projection space selected with:

```
--dim
```

or in all available dimensions when:

```
--dim 0
```

is used.

---

## Structural processing

For each density-defined cluster, `clusterMD` performs structural processing using the original trajectory and topology.

The workflow includes:

1. identification of cluster-member trajectory frames,
2. selection of the representative frame,
3. extraction of cluster-member frames,
4. structural fitting using `carma`,
5. average-structure calculation,
6. superposition generation,
7. RMSD evaluation,
8. optional RMSD-based structural filtering,
9. recalculation of the fitted structural ensemble after filtering.

Structural processing is downstream of clustering.

The original HDBSCAN labels and cluster populations are not altered by the structural filtering stage.

---

## RMSD-based structural filtering

When:

```
--rmsd > 0
```

is used, `clusterMD` applies downstream structural filtering.

The general process is:

1. structurally fit the cluster ensemble,
2. calculate RMSD relative to the fitted ensemble average,
3. retain structures satisfying the selected RMSD criterion,
4. rebuild the filtered structural ensemble from the original retained trajectory frames,
5. recalculate the final fitted ensemble and average structure.

Because the structural average is recalculated after filtering, the final reported maximum RMSD to the final average can differ from the filtering threshold used during the preceding filtering step.

This does not imply that HDBSCAN assignments were changed.

---

## Structural reporting

For structurally processed clusters, reporting includes values such as:

```
HDBSCAN Frames
Structural Frames
Retained %
Representative Frame
Max RMSD to avg
```

These values separate clustering membership from downstream structural retention.

---

## Observed label changes

`clusterMD` examines the chronological sequence of assigned cluster labels.

It can report observed changes between assigned labels and their frequencies.

These outputs are intended as descriptive summaries of trajectory ordering.

They should not be interpreted as direct estimates of:

- transition rates,
- kinetic constants,
- Markov-model probabilities,
- metastable-state lifetimes,
- or mechanistic pathway probabilities.

Frames labeled:

```
-1
```

remain explicitly unassigned/noise and are handled according to the implemented chronological change-counting semantics.

---

## Post-run diagnostic mode

`clusterMD` can inspect an existing cluster-assignment file from inside a previous results directory.

For example:

```
clusterMD --transitions
```

can be used inside a `clusterMD_results_*` directory to inspect observed label changes without rerunning the full clustering pipeline.

A specific cluster pair can be examined with:

```
clusterMD --transitions 2 7
```

---

## Output directory

Each full run creates a timestamped directory using the pattern:

```
clusterMD_results_DD.MM.YYYY_HH.MM.SS.mmm/
```

Millisecond timestamps reduce the possibility of collisions between rapidly repeated runs.

---

## Typical outputs

Depending on the selected options and available structural inputs, outputs may include:

- clustered PCA/dPCA coordinates
- HDBSCAN assignment tables
- cluster population summaries
- cluster projection figures
- population-derived free-energy landscapes
- observed-label-change summaries
- observed-label-change plots
- representative PDB structures
- cluster-member structural trajectories
- structurally fitted trajectories
- structural superpositions
- average structures
- RMSD reports
- execution logs
- provenance reports
- software-version information

Not every output is produced in every execution mode.

---

## Provenance reporting

`clusterMD` records runtime provenance information including software versions where available.

The report includes information such as:

```
clusterMD version
Python version
fast_hdbscan version
NumPy version
SciPy version
carma version
```

The exact command invocation is also preserved in the execution record.

---

## File discovery behavior

`clusterMD` attempts to locate suitable PCA/dPCA projection files and structural files automatically.

When multiple candidate files are found, the program:

- emits a warning,
- selects according to its deterministic file-selection logic,
- records the selected input in the execution log.

Users should verify that the selected files correspond to the intended analysis system.

---

## Validation

The current software version has undergone validation using synthetic and real molecular-dynamics datasets.

The validation program has included:

- frame-ID preservation
- projection stride accounting
- cluster population accounting
- deterministic relabeling checks
- preservation of the `-1` unassigned/noise label
- observed-label-change counting checks
- representative-frame verification
- structural extraction verification
- output/report consistency checks
- end-to-end PCA/dPCA workflow testing
- controlled synthetic clustering tests
- HDBSCAN parameter-sensitivity analysis
- retained-dimensionality testing
- stride-sensitivity testing
- structural representative comparisons
- RMSD-filtering validation
- tests confirming that structural filtering does not alter HDBSCAN assignments
- malformed-input handling
- missing-input handling
- NaN/Inf validation
- one-dimensional-input handling
- all-noise and single-cluster edge cases
- invalid-parameter handling
- multiple-input-file ambiguity handling
- performance and scaling measurements
- deterministic same-environment reruns
- real-system regression testing

---

## Validation environment

The current validated Python environment is:

```
Python          3.12.3
cloudpickle     3.1.2
contourpy       1.3.3
cycler          0.12.1
fast_hdbscan    0.3.2
fonttools       4.66.1
joblib          1.6.0
kiwisolver      1.5.1
llvmlite        0.50.0
matplotlib      3.11.2
narwhals        2.26.0
networkx        3.7
numba           0.68.0
numpy           1.26.4
packaging       26.3
pillow          12.3.0
psutil          7.2.2
pyparsing       3.3.3
python-dateutil 2.9.0.post0
scikit-learn    1.9.1
scipy           1.17.1
six             1.17.0
threadpoolctl   3.7.0
```

The exact environment is stored in:

```
requirements-lock.txt
```

The validation environment also used:

```
carma 2.3
```

---

## Current validated version

```
clusterMD 1.1.0
```

This version identifier is recorded in the software provenance report.

---

## Reproducibility

For maximum reproducibility:

1. use the same `clusterMD` version,
2. reproduce the Python environment using `requirements-lock.txt`,
3. use the same `carma` version where structural processing is required,
4. use identical PCA/dPCA input data,
5. use the same trajectory and topology files,
6. preserve the same clustering parameters,
7. preserve the same dimensionality and stride settings,
8. preserve the same structural-processing options.

Example:

```
python -m pip install -r requirements-lock.txt
```

followed by the same `clusterMD` invocation used in the original analysis.

---

## Scientific scope

`clusterMD` is intended to support reproducible exploration of conformational organization in multidimensional molecular-dynamics projection spaces.

Appropriate uses include:

- density-defined clustering of PCA/dPCA projections,
- descriptive comparison of cluster populations,
- extraction of representative observed structures,
- inspection of structurally fitted cluster ensembles,
- descriptive free-energy visualization,
- chronological inspection of changes between assigned cluster labels,
- parameter-sensitivity and structural-validation workflows.

The software does not replace dedicated kinetic modeling, Markov state modeling, free-energy convergence analysis, or independent structural validation where those are required by the scientific question.

---

## Known interpretation boundaries

Users should keep the following distinctions explicit:

### Cluster does not automatically mean metastable state

HDBSCAN detects density structure in the selected projection space.

Additional evidence is required before assigning a kinetic or thermodynamic state interpretation.

### Observed label change does not mean transition rate

A chronological change from one assigned cluster label to another is an observed label change.

It is not, by itself, a kinetic-rate measurement.

### Representative does not mean medoid

The representative structure is selected as the observed frame nearest the arithmetic centroid in retained PCA/dPCA space.

### Structural filtering does not redefine clusters

RMSD filtering affects downstream structural ensembles only.

It does not alter the original HDBSCAN cluster labels or populations.

### 2D visualization does not define the clustering space

Displayed 2D figures are visual projections.

Clustering is performed in the selected multidimensional space.

---

## Development status

`clusterMD` is under active research-software development.

The current development focus includes:

- release packaging,
- dependency and environment reproducibility,
- documentation,
- structural-output consistency,
- reporting improvements,
- and preparation of the accompanying Application Note.

The validated clustering and structural-analysis core is being preserved while release-oriented improvements are added around it.

---

## Author

**Anastasios Melas**

MSc Applied Bioinformatics and Data Analysis\
Democritus University of Thrace

---

## Repository

[github.com/tasosmelas/md-hdbscan-clustering](https://github.com/tasosmelas/md-hdbscan-clustering)
