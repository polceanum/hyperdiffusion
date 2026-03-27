#!/usr/bin/env bash
set -e

PY=${PY:-/usr/local/Caskroom/miniforge/base/envs/aurel/bin/python}

if [[ "$1" == "clean" ]]; then
  echo "[paper] cleaning temporary files"
  rm -f paper/*.aux paper/*.log paper/*.out paper/*.fdb_latexmk paper/*.fls paper/*.synctex.gz paper/*.toc paper/*.lof paper/*.lot
  rm -f paper/*.pdf
  rm -f paper/figures/plots/*
  rm -f paper/sections/annex.tex
  echo "[paper] cleaned"
  exit 0
fi

"$PY" paper/tables/gen_tables.py
"$PY" paper/figures/gen_plots.py

cd paper
export PATH="/usr/local/texlive/2026/bin/universal-darwin:$PATH"
/usr/local/texlive/2026/bin/universal-darwin/latexmk -pdf paper.tex
/usr/local/texlive/2026/bin/universal-darwin/latexmk -c paper.tex

echo "[paper] built paper.pdf (aux files cleaned)"
