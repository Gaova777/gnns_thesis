#!/usr/bin/env bash
# Regenera tablas, figuras y PDF del manuscrito v4, una etapa tras otra.
# Uso: tesis_v4/scripts/construir.sh [PID a esperar]
set -u
cd "$(dirname "$0")/../.."
if [ -n "${1:-}" ]; then while kill -0 "$1" 2>/dev/null; do sleep 20; done; fi
L=tesis_v4/construir.log
: > $L
~/.local/bin/uv run --frozen python tesis_v4/scripts/analisis.py >> $L 2>&1; echo "analisis rc=$?" >> $L
~/.local/bin/uv run --frozen python tesis_v4/scripts/figuras.py >> $L 2>&1; echo "figuras rc=$?" >> $L
B=$(ls -d ~/.TinyTeX/bin/*)
cd tesis_v4
$B/pdflatex -interaction=nonstopmode main.tex > latex1.log 2>&1
$B/biber main > biber.log 2>&1
$B/pdflatex -interaction=nonstopmode main.tex > latex2.log 2>&1
$B/pdflatex -interaction=nonstopmode main.tex > latex3.log 2>&1
echo "latex rc=$? paginas=$(grep -o 'Output written.*' latex3.log)" >> construir.log
echo "fin construir" >> construir.log
