#!/usr/bin/env bash
# Creates a readable commit history (one commit per concern) using YOUR git identity.
# Usage:  git init -b main && git config user.name "Your Name" && git config user.email "you@example.com" && bash scripts/first_commits.sh
set -euo pipefail
c() { msg="$1"; shift; git add -- "$@"; git commit -q -m "$msg"; echo "committed: $msg"; }
c "Add project scaffold: README, licence, packaging, CI, data notes" README.md LICENSE .gitignore pyproject.toml requirements.txt .github data/README.md
c "Add configuration and data pipeline (PNG discovery, validation, depth-hole filling, synthetic test data)" src/treeskel/__init__.py src/treeskel/config.py src/treeskel/data.py src/treeskel/dataset.py src/treeskel/synthetic.py
c "Add verified numerical methods (Newton, finite differences, ODE solvers, quadrature, ridge solvers)" src/treeskel/numerics.py src/treeskel/skeleton.py src/treeskel/postprocess.py
c "Add regression models, CNN, features and metrics" src/treeskel/models.py src/treeskel/cnn.py src/treeskel/features.py src/treeskel/metrics.py
c "Add 3D lifting, ODE-based alternative and training/inference pipeline" src/treeskel/geometry.py src/treeskel/ode_recon.py src/treeskel/pipeline.py src/treeskel/cli.py
c "Add experiments for validation, optimisation and performance analysis" src/treeskel/experiments.py scripts/run_experiments.py
c "Add pytest suite" tests
c "Add Colab runner notebook" notebooks
c "Add generated test report and API documentation" docs
c "Add AI-use disclosure template" AI_USE.md scripts/first_commits.sh
git log --oneline
