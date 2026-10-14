#!/usr/bin/env bash
# Medidas robustas y control de pesos al azar para la corrida sin el cierre (THE-44, THE-46).
# Una tarea tras otra, nunca en paralelo.
set -u
cd "$(dirname "$0")/../.."
L=results_v4_precierre/robusta
~/.local/bin/uv run --frozen python scripts/v4/reunion_0710/metrica_robusta.py --variante precierre > $L/metrica_robusta.log 2>&1
echo "fin robusta rc=$?" >> $L/metrica_robusta.log
~/.local/bin/uv run --frozen python scripts/v4/reunion_0710/control_aleatorio.py --variante precierre > $L/control_aleatorio.log 2>&1
echo "fin control rc=$?" >> $L/control_aleatorio.log
