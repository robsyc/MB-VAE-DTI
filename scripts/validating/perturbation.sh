#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/MB-VAE-DTI"
SCRIPT="$ROOT/mb_vae_dti/validating/perturbation.py"
RESULTS_DIR="$ROOT/data/results/perturbation"
MODEL="multi_modal"
STEPS=11

mkdir -p "$RESULTS_DIR"

datasets=(DAVIS KIBA)
splits=(split_rand split_cold)
branches=(drug target)
drug_features=(EMB-BiomedGraph EMB-BiomedImg EMB-BiomedText)
target_features=(EMB-ESM EMB-NT)

slugify() {
  local s="$1"
  s="${s,,}"                 # lowercase
  s="${s// /_}"              # spaces -> _
  s="${s//-/_}"              # hyphens -> _
  s="$(echo "$s" | tr -cs 'a-z0-9_.' '_')"  # non-alnum -> _
  echo "$s"
}

split_short() {
  echo "${1#split_}"
}

for dataset in "${datasets[@]}"; do
  dslug="$(slugify "$dataset")"
  for split in "${splits[@]}"; do
    sshort="$(split_short "$split")"
    for branch in "${branches[@]}"; do
      # Run without specifying any feature
      outfile="$RESULTS_DIR/${MODEL}_${dslug}_${sshort}_${branch}.json"
      echo "Running: dataset=$dataset split=$split branch=$branch (no feature) -> $outfile"
      python "$SCRIPT" \
        --dataset "$dataset" --split "$split" --model "$MODEL" --branch "$branch" \
        --steps "$STEPS" --output "$outfile"

      # Feature-specific runs for this branch
      if [[ "$branch" == "drug" ]]; then
        feats=("${drug_features[@]}")
      else
        feats=("${target_features[@]}")
      fi

      for feat in "${feats[@]}"; do
        fslug="$(slugify "$feat")"
        outfile="$RESULTS_DIR/${MODEL}_${dslug}_${sshort}_${branch}_${fslug}.json"
        echo "Running: dataset=$dataset split=$split branch=$branch feature=$feat -> $outfile"
        python "$SCRIPT" \
          --dataset "$dataset" --split "$split" --model "$MODEL" --branch "$branch" \
          --feature "$feat" --steps "$STEPS" --output "$outfile"
      done
    done
  done
done