#!/usr/bin/env bash
# build_pdf.sh — WRITEUP.md -> paper.pdf (to read) + paper.tex (arXiv-portable source).
# Approach: pandoc builds the LaTeX structure; we then map Unicode math symbols to
# LaTeX math IN THE GENERATED .tex (so there is no markdown math-parser to fight),
# and compile with pdflatex using only standard packages (what arXiv expects).
set -euo pipefail
cd "$(dirname "$0")"

# 1. body = from the Abstract heading onward (title/author come from metadata)
awk '/^## Abstract/{f=1} f' WRITEUP.md > .paper_body.md

# 2. metadata header (literal; body appended separately so backticks/$ are safe)
cat > .paper_full.md <<'YAML'
---
title: "Conversational Domain Adaptation of IndicTrans2 across 21 Indic Languages via Experience Replay and Model Soups"
author: "Aditya Pratap Singh (Independent Researcher)"
date: "June 2026"
geometry: margin=1in
fontsize: 11pt
colorlinks: true
linkcolor: blue
urlcolor: blue
---

YAML
cat .paper_body.md >> .paper_full.md

# 3. pandoc -> standalone LaTeX
pandoc .paper_full.md -s -o paper.tex

# 4. map Unicode math -> LaTeX math in the generated .tex (single backslashes via \\)
sed -i 's/−/$-$/g; s/·/$\\cdot$/g; s/θ/$\\theta$/g; s/α/$\\alpha$/g; s/Δ/$\\Delta$/g; s/≈/$\\approx$/g; s/≥/$\\ge$/g; s/≤/$\\le$/g; s/±/$\\pm$/g; s/→/$\\to$/g; s/↔/$\\leftrightarrow$/g; s/×/$\\times$/g; s/↑/$\\uparrow$/g; s/↓/$\\downarrow$/g' paper.tex

# 5. compile (twice: settle any refs); quiet unless it errors
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/tmp/ptex.log 2>&1 || { tail -25 /tmp/ptex.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/tmp/ptex.log 2>&1 || { tail -25 /tmp/ptex.log; exit 1; }

# 6. flag any Unicode we missed (would render as a blank box)
if grep -nP '[^\x00-\x7F]' paper.tex | grep -vP '[–—‘’“”…]' >/dev/null; then
  echo "WARN: non-ASCII chars remain in paper.tex (check rendering):"
  grep -nP '[^\x00-\x7F]' paper.tex | grep -vP '[–—‘’“”…]' | head
fi

rm -f .paper_body.md .paper_full.md paper.aux paper.log paper.out
echo "OK -> paper.pdf  +  paper.tex"
