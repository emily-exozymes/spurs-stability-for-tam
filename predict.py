#!/usr/bin/env python
"""
SPURS-Stability: predict mutation-induced ddG for a protein.

Strategy: parse the user's PDB once with each model's config (single and multi
use slightly different alphabets / featurization). Walk the mutations CSV and
route each row:
  - 1 mutation token (e.g. 'W1A')      -> single model
  - 2+ mutation tokens (e.g. 'V2C:P3T') -> multi model
Output a CSV with mutant + ddG.
"""

import os
import re
import sys
import glob
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# ---------- Inputs ----------
CHAIN    = os.environ.get("chain", "A").strip() or "A"
PDB_NAME = os.environ.get("pdb_name", "input").strip() or "input"

os.makedirs("out", exist_ok=True)

# ---------- Locate inputs ----------
pdb_candidates = sorted(glob.glob("inputs/*.pdb"))
if not pdb_candidates:
    sys.exit("ERROR: no .pdb file found in inputs/")
pdb_path = pdb_candidates[0]
print(f"PDB: {pdb_path}")

csv_candidates = sorted(glob.glob("inputs/*.csv"))
if not csv_candidates:
    sys.exit("ERROR: no mutations CSV found in inputs/")
mut_csv = csv_candidates[0]
print(f"Mutations CSV: {mut_csv}")

df = pd.read_csv(mut_csv)
if "mutant" not in df.columns:
    sys.exit("ERROR: mutations CSV must have a 'mutant' column")
print(f"Loaded {len(df)} mutations")

# ---------- GPU check ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("WARNING: no GPU - running on CPU (slow).")
else:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------- Load models ----------
from spurs.inference import (
    get_SPURS_from_hub,
    get_SPURS_multi_from_hub,
    parse_pdb,
    parse_pdb_for_mutation,
)

# Classify rows up front so we only load the model(s) we actually need.
def split_mut_tokens(s):
    return [t for t in re.split(r"[,:;]", str(s).strip()) if t]

df["tokens"] = df["mutant"].apply(split_mut_tokens)
df["n_mut"] = df["tokens"].apply(len)

has_single = (df["n_mut"] == 1).any()
has_multi  = (df["n_mut"] >= 2).any()
if (df["n_mut"] == 0).any():
    bad = df[df["n_mut"] == 0]
    print(f"WARNING: {len(bad)} rows have empty mutant strings; they will be skipped.")

model_single = cfg_single = pdb_single = None
model_multi  = cfg_multi  = pdb_multi  = None

if has_single:
    print("Loading SPURS single model...")
    model_single, cfg_single = get_SPURS_from_hub(device=str(device))
    model_single.eval()
    pdb_single = parse_pdb(pdb_path, PDB_NAME, CHAIN, cfg_single, device=str(device))
    print(f"  WT length: {len(pdb_single['seq'])}")

if has_multi:
    print("Loading SPURS multi model...")
    model_multi, cfg_multi = get_SPURS_multi_from_hub(device=str(device))
    model_multi.eval()
    pdb_multi = parse_pdb(pdb_path, PDB_NAME, CHAIN, cfg_multi, device=str(device))

# ---------- Single-mutation scoring ----------
def parse_single_mut(tok):
    m = re.match(r"^([A-Za-z])(\d+)([A-Za-z])$", tok.strip())
    if not m:
        raise ValueError(f"Bad mutation token: {tok!r}")
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper()

ddg_out = [None] * len(df)

if has_single:
    with torch.no_grad():
        # SPURS single model returns a [L, 20] matrix when return_logist=True.
        ddg_matrix = model_single(pdb_single, return_logist=True).detach().cpu().numpy()
    print(f"Single ddG matrix shape: {ddg_matrix.shape}")

    seq = pdb_single["seq"]  # WT sequence as a string
    for i, row in df.iterrows():
        if row["n_mut"] != 1:
            continue
        try:
            orig, pos1, new = parse_single_mut(row["tokens"][0])
            pos0 = pos1 - 1
            if pos0 < 0 or pos0 >= len(seq):
                raise ValueError(f"Position {pos1} out of range for sequence length {len(seq)}")
            if seq[pos0] != orig:
                print(f"  WARNING: row {i} mutant {row['mutant']}: expected {orig} at position {pos1}, found {seq[pos0]} - proceeding anyway")
            ddg_out[i] = float(ddg_matrix[pos0, ALPHABET.index(new)])
        except Exception as e:
            print(f"  row {i} ({row['mutant']}) failed: {e}")
            ddg_out[i] = float("nan")

# ---------- Multi-mutation scoring ----------
if has_multi:
    multi_rows = df[df["n_mut"] >= 2]
    mut_info_list = [list(toks) for toks in multi_rows["tokens"].tolist()]

    try:
        mut_ids, append_tensors = parse_pdb_for_mutation(mut_info_list)
        pdb_multi["mut_ids"] = mut_ids
        pdb_multi["append_tensors"] = append_tensors.to(device)

        with torch.no_grad():
            ddg_multi = model_multi(pdb_multi).detach().cpu().numpy()

        for slot, (i, _) in enumerate(multi_rows.iterrows()):
            try:
                ddg_out[i] = float(ddg_multi[slot])
            except Exception as e:
                print(f"  multi row {i} failed: {e}")
                ddg_out[i] = float("nan")
    except Exception as e:
        print(f"ERROR in multi-mutation scoring: {e}")
        for i, _ in multi_rows.iterrows():
            ddg_out[i] = float("nan")

# ---------- Save ----------
df["ddG"] = ddg_out
out_df = df.drop(columns=["tokens", "n_mut"])
out_df.to_csv("out/predictions.csv", index=False)

n_done = sum(1 for v in ddg_out if v is not None and not (isinstance(v, float) and np.isnan(v)))
print(f"\nWrote out/predictions.csv: {n_done}/{len(df)} mutations scored successfully.")
