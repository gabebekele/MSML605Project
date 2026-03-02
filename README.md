Project Overview

Milestone 1 implements a deterministic pipeline for face verification experiments using the LFW dataset, including ingestion, pair generation, and similarity benchmarking. Ingestion downloads or loads LFW, applies a fixed train/validation/test split, and writes a manifest summarizing the dataset splits. Pair generation creates positive and negative verification pairs for each split, saved as CSV files. The similarity benchmark computes cosine similarity and Euclidean distance on synthetic vectors, comparing Python loop and vectorized implementations for speed and correctness. This milestone ensures reproducibility: running the scripts with the same configuration produces identical outputs.

Repository Layout:
The repository is organized with src/ containing the Python package code, scripts/ holding command-line scripts, and configs/ including YAML configuration files specifying seeds, paths, and split/pair policies. Generated artifacts such as manifests, pairs, and benchmark results are saved under outputs/ and are not committed. The data/ directory contains the dataset cache and is also ignored. Optional tests and determinism checks can be found in tests/. Committed files include the source code, scripts, configurations, README, and requirements; outputs and dataset caches are ignored.

How to Run:
To reproduce the milestone, first clone the repository and set up a Python virtual environment:

git clone https://github.com/gabebekele/msml605project <new_repo_name>
cd <new_repo_name>
python3 -m venv .venv

Activate the virtual environment:

for macOS/Linux
source .venv/bin/activate

for Windows
.venv\Scripts\Activate.ps1

And then install the requirements:

pip install -r requirements.txt

Next, run deterministic ingestion with:

python scripts/ingest.py --config configs/m1.yaml

This produces outputs/manifest.json, which summarizes the dataset splits, counts, seed, and split policy. Then, generate verification pairs using:

python scripts/make_pairs.py --config configs/m1.yaml

This produces outputs/pairs/train.csv, outputs/pairs/val.csv, and outputs/pairs/test.csv. Finally, run the similarity benchmark with:

python -m scripts.bench_similarity

The benchmark prints timing and correctness results for Python loop versus vectorized implementations, and saves them to outputs/bench_results.txt.


Outputs:
The outputs of this milestone include the manifest file summarizing splits and counts, three CSV files containing verification pairs for train, validation, and test sets, and benchmark results for similarity computations. All outputs are stored under the outputs/ directory.

Determinism Notes:
A fixed seed (123) is used to ensure determinism. The pipeline is deterministic because identities and filenames are sorted before splitting, all random operations use a fixed seed via np.random.default_rng(seed), and the pair generation policy is consistently applied. Rerunning ingestion and pair generation with the same configuration in order to produce identical outputs.
