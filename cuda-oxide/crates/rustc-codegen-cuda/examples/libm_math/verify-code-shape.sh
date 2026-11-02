#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
ptx="${1:-$root/libm_math.ptx}"

if [[ ! -f "$ptx" ]]; then
  echo "PTX file not found: $ptx" >&2
  exit 1
fi

python3 - "$ptx" <<'PY'
import re
import sys

ptx_path = sys.argv[1]
ptx = open(ptx_path, encoding="utf-8").read()
# Strip comments so a mention in a comment cannot satisfy or hide a check.
ptx = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)
local_memory = re.compile(r"(?m)(?:^|\s)\.local\b|ld\.local|st\.local")


def entry_body(name):
    header = re.search(r"\.entry\s+" + re.escape(name) + r"\s*\(", ptx)
    if header is None:
        sys.exit(f"error: missing PTX entry {name} in {ptx_path}")
    brace = ptx.find("{", header.end())
    if brace < 0:
        sys.exit(f"error: entry {name} has no body")
    depth = 1
    end = brace + 1
    while depth and end < len(ptx):
        depth += (ptx[end] == "{") - (ptx[end] == "}")
        end += 1
    if depth:
        sys.exit(f"error: unterminated PTX entry {name}")
    return ptx[brace + 1 : end - 1]


# Counted loops that index a per-thread array and call libdevice's erff. A
# `.local` depot means LLVM left the loop rolled: for `erf_fill_loop`, `opt`
# priced the libdevice body before `__nvvm_reflect` folded (no `-mcpu`); for
# `erf_pair_loop`, it ran O2's full-unroll threshold instead of O3's.
for name in ("erf_fill_loop", "erf_pair_loop"):
    if local_memory.search(entry_body(name)):
        sys.exit(f"error: {name} uses local memory in {ptx_path}")
    print(f"{name}: no local memory PASS")

# libm's arch-dispatched sqrt reaches libdevice's correctly rounded sqrt.
for instruction in ("sqrt.rn.f32", "sqrt.rn.f64"):
    if instruction not in entry_body("libm_f32") + entry_body("libm_f64"):
        sys.exit(f"error: no {instruction} in the libm kernels of {ptx_path}")
    print(f"{instruction}: PASS")
print("libm_math PTX shape: PASS")
PY
