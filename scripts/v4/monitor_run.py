#!/usr/bin/env python
"""Monitor de corridas v4: lee <log-dir>/progress.jsonl y HEARTBEAT_* (src/monitoring/progress.py).

Por etapa (train / explain / synthetic / ...): hechos/total de la invocación en curso (desde su
último evento «plan»), acumulado de todo el archivo, errores con mensaje, tiempo medio por
unidad y ETA. Luego un bloque de SEÑALES DE ALERTA:

  CRÍTICAS (exit 2)
  - cuelgue: HEARTBEAT_<etapa> sin actualizar > --stale-min (45 min) con la etapa sin terminar.
    Un config de Optuna o un explicador sobre un nodo de alto grado tardan minutos, no 45: si el
    latido (que se refresca en cada evento y, en explain, en cada nodo) no se mueve, está colgado.
  - ≥ --oom-streak (2) errores OOM seguidos en una etapa: el siguiente también fallará; hay que
    bajar hidden_dim/nodos o trocear por proceso antes de seguir quemando GPU.
  - compuerta: tras ≥ --gate-min-n (10) configuraciones entrenadas, tasa de gate_passed <
    --gate-min-rate (30 %). En v3 pasaba 23/60 ≈ 38 % CON las etiquetas mal; con etiquetas
    correctas debe pasar más. Menos del 30 % indica datos o entrenamiento rotos.
  - mediana de val_pr_auc (etapa train) fuera de [--prauc-lo, --prauc-hi] = [0,20 ; 0,95]. Con
    la prevalencia de val (914/9.983 ≈ 0,09) un clasificador al azar da PR-AUC ≈ 0,09; los
    modelos de la literatura sobre Elliptic dan ≈ 0,4-0,8. < 0,20 = no aprende (o etiquetas
    mal); > 0,95 = fuga de información o etiquetas triviales.
  - NaN en métricas primarias (val_pr_auc / test_pr_auc en train y synthetic; métrica primaria
    del explicador en explain: spearman_full para GNNExplainer/ShapleyFeatures, spearman_edges
    para PGExplainer).
  - explicador con > --noapl-max (50 %) de unidades «no_aplica» (métrica primaria NaN o
    n_measurable = 0) cuando por diseño SÍ aplica (la NaN de spearman_full en PGExplainer es
    esperada y no cuenta). En synthetic se lee <explicador>_status.
  AVISOS (exit 1): errores aislados (no OOM), latido > 15 min.

Uso:
  uv run --frozen python scripts/v4/monitor_run.py --once            # un informe y sale
  uv run --frozen python scripts/v4/monitor_run.py --watch 10        # refresca cada 10 min
  uv run --frozen python scripts/v4/monitor_run.py --once --json     # JSON para otro agente
Exit: 0 sin alertas · 1 solo avisos · 2 alguna alerta crítica.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PRIMARY_EXPLAIN = {"GNNExplainer": "spearman_full", "ShapleyFeatures": "spearman_full",
                   "GNNShap": "spearman_full", "PGExplainer": "spearman_edges"}


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def load_events(path: Path) -> list[dict]:
    ev = []
    if not path.exists():
        return ev
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                ev.append({"event": "corrupt", "stage": "?", "msg": line[:120]})
    return ev


def is_oom(msg: str) -> bool:
    m = (msg or "").lower()
    return "oom" in m or "out of memory" in m or "outofmemory" in m


def explainer_of(run_id: str) -> str | None:
    return run_id.rsplit("__", 1)[1] if "__" in run_id else None


def stage_report(stage: str, evs: list[dict], a) -> dict:
    last_plan = max((i for i, e in enumerate(evs) if e.get("event") == "plan"), default=-1)
    cur = evs[last_plan + 1:] if last_plan >= 0 else evs
    total = evs[last_plan].get("total") if last_plan >= 0 else None
    ends_cur = [e for e in cur if e.get("event") == "end"]
    skips_cur = [e for e in cur if e.get("event") == "skip"]
    errs_cur = [e for e in cur if e.get("event") == "error"]
    finished_ids = {e.get("run_id") for e in ends_cur + skips_cur + errs_cur}
    started = [e for e in cur if e.get("event") == "start"]
    running = [e.get("run_id") for e in started if e.get("run_id") not in finished_ids]
    done = len(finished_ids)
    durs = [e["duration_s"] for e in ends_cur if isinstance(e.get("duration_s"), (int, float))]
    mean_s = st.mean(durs) if durs else None
    remaining = (total - done) if isinstance(total, int) else None
    eta_s = remaining * mean_s if (remaining is not None and mean_s) else None

    all_ends = [e for e in evs if e.get("event") == "end"]
    all_errs = [e for e in evs if e.get("event") == "error"]
    rep = dict(stage=stage, total=total, done=done, ok=len(ends_cur), skipped=len(skips_cur),
               errors=len(errs_cur), running=running[-3:], mean_unit_s=round(mean_s, 1) if mean_s else None,
               eta_h=round(eta_s / 3600, 2) if eta_s is not None else None,
               cumulative_end=len({e.get("run_id") for e in all_ends}),
               cumulative_errors=len(all_errs),
               last_errors=[{"run_id": e.get("run_id"), "msg": (e.get("msg") or "")[:240], "ts": e.get("ts")}
                            for e in all_errs[-5:]],
               finished=(isinstance(total, int) and done >= total))

    alerts = []
    # --- OOM seguidos (en orden de eventos terminales) ---
    streak = best = 0
    for e in evs:
        if e.get("event") == "error":
            streak = streak + 1 if is_oom(e.get("msg", "")) else 0
        elif e.get("event") == "end":
            streak = 0
        best = max(best, streak)
    rep["max_oom_streak"] = best
    rep["current_oom_streak"] = streak
    if streak >= a.oom_streak:
        alerts.append(("critical", f"{stage}: {streak} OOM seguidos (último: {rep['last_errors'][-1]['run_id']})"))
    elif best >= a.oom_streak:
        alerts.append(("warning", f"{stage}: hubo una racha de {best} OOM seguidos (ya se recuperó)"))
    non_oom = [e for e in all_errs if not is_oom(e.get("msg", ""))]
    if non_oom:
        alerts.append(("warning", f"{stage}: {len(non_oom)} error(es) no-OOM; último: "
                                  f"{non_oom[-1].get('run_id')}: {(non_oom[-1].get('msg') or '')[:160]}"))

    metrics = [e.get("metrics") or {} for e in all_ends]
    # --- compuerta y PR-AUC (train) ---
    if stage == "train":
        gates = [m.get("gate_passed", m.get("quality_passed")) for m in metrics]
        gates = [bool(g) for g in gates if g is not None]
        rep["gate_n"], rep["gate_passed"] = len(gates), sum(gates)
        if len(gates) >= a.gate_min_n:
            rate = sum(gates) / len(gates)
            rep["gate_rate"] = round(rate, 3)
            if rate < a.gate_min_rate:
                alerts.append(("critical", f"train: tasa de compuerta {rate:.0%} ({sum(gates)}/{len(gates)}) "
                                           f"< {a.gate_min_rate:.0%}"))
        vals = [m.get("val_pr_auc") for m in metrics]
        good = [v for v in vals if not _isnan(v)]
        if good:
            med = st.median(good)
            rep["val_pr_auc_median"] = round(med, 4)
            if len(good) >= a.prauc_min_n and not (a.prauc_lo <= med <= a.prauc_hi):
                alerts.append(("critical", f"train: mediana val_pr_auc {med:.3f} fuera de "
                                           f"[{a.prauc_lo}, {a.prauc_hi}] (n={len(good)})"))
    # --- NaN en métricas primarias ---
    nan_units = []
    for e in all_ends:
        m = e.get("metrics") or {}
        rid = e.get("run_id", "")
        if stage in ("train", "synthetic"):
            keys = [k for k in ("val_pr_auc", "test_pr_auc") if k in m]
        elif stage == "explain":
            ex = explainer_of(rid)
            keys = [PRIMARY_EXPLAIN[ex]] if ex in PRIMARY_EXPLAIN and PRIMARY_EXPLAIN[ex] in m else []
        else:
            keys = [k for k, v in m.items() if isinstance(v, float)]
        if any(_isnan(m.get(k)) for k in keys):
            nan_units.append(rid)
    rep["nan_primary_units"] = nan_units[-10:]
    # En explain, la NaN primaria ES «no_aplica»: se evalúa por la regla de proporción de abajo
    # (una NaN aislada puede ser legítima: 0 nodos medibles en una celda con pocos VP).
    if nan_units and stage != "explain":
        alerts.append(("critical", f"{stage}: {len(nan_units)} unidad(es) con NaN en métrica primaria "
                                   f"(p. ej. {nan_units[-1]})"))
    # --- no_aplica por explicador ---
    noapl: dict[str, list[int]] = {}
    for e in all_ends:
        m = e.get("metrics") or {}
        if stage == "explain":
            ex = explainer_of(e.get("run_id", ""))
            if ex not in PRIMARY_EXPLAIN:
                continue
            bad = _isnan(m.get(PRIMARY_EXPLAIN[ex])) or m.get("n_measurable") == 0
            noapl.setdefault(ex, [0, 0]); noapl[ex][0] += int(bad); noapl[ex][1] += 1
        elif stage == "synthetic":
            for k, v in m.items():
                if k.endswith("_status"):
                    ex = k[:-7]
                    noapl.setdefault(ex, [0, 0]); noapl[ex][0] += int(v != "ok"); noapl[ex][1] += 1
    rep["no_aplica"] = {ex: f"{b}/{n}" for ex, (b, n) in noapl.items()}
    for ex, (b, n) in noapl.items():
        if n >= a.noapl_min_n and b / n > a.noapl_max:
            alerts.append(("critical", f"{stage}: {ex} con {b}/{n} ({b/n:.0%}) unidades no_aplica/NaN "
                                       f"> {a.noapl_max:.0%}"))
    rep["alerts"] = alerts
    return rep


def heartbeat_alerts(log_dir: Path, reports: dict, a) -> tuple[dict, list]:
    hb, alerts = {}, []
    now = datetime.now(timezone.utc)
    for p in sorted(log_dir.glob("HEARTBEAT_*")):
        if p.suffix == ".tmp":
            continue
        stage = p.name[len("HEARTBEAT_"):]
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            age = (now - _parse_ts(d["ts"])).total_seconds() / 60
        except Exception:  # noqa: BLE001
            d, age = {}, (time.time() - p.stat().st_mtime) / 60
        hb[stage] = {"age_min": round(age, 1), "run_id": d.get("run_id"), "note": d.get("note")}
        rep = reports.get(stage)
        if rep and rep.get("finished"):
            continue
        if age > a.stale_min:
            alerts.append(("critical", f"{stage}: latido sin actualizar hace {age:.0f} min (> {a.stale_min}) "
                                       f"— posible cuelgue en {d.get('run_id')!r}"))
        elif age > a.stale_warn_min:
            alerts.append(("warning", f"{stage}: latido de hace {age:.0f} min (en {d.get('run_id')!r})"))
    return hb, alerts


def build(a) -> dict:
    log_dir = Path(a.log_dir)
    evs = load_events(log_dir / "progress.jsonl")
    by_stage: dict[str, list] = {}
    for e in evs:
        by_stage.setdefault(e.get("stage", "?"), []).append(e)
    reports = {s: stage_report(s, ev, a) for s, ev in by_stage.items()}
    hb, hb_alerts = heartbeat_alerts(log_dir, reports, a)
    alerts = [al for r in reports.values() for al in r.pop("alerts")] + hb_alerts
    corrupt = sum(1 for e in evs if e.get("event") == "corrupt")
    if corrupt:
        alerts.append(("warning", f"{corrupt} línea(s) ilegibles en progress.jsonl"))
    level = 2 if any(s == "critical" for s, _ in alerts) else (1 if alerts else 0)
    return dict(ts=datetime.now(timezone.utc).isoformat(timespec="seconds"), log_dir=str(log_dir),
                n_events=len(evs), stages=reports, heartbeats=hb,
                alerts=[{"level": s, "msg": m} for s, m in alerts], exit_code=level)


def render(r: dict) -> str:
    L = [f"== monitor v4 · {r['ts']} · {r['log_dir']} · {r['n_events']} eventos =="]
    if not r["n_events"]:
        L.append("  (sin eventos todavía)")
    for s, x in r["stages"].items():
        tot = x["total"] if x["total"] is not None else "?"
        pct = f" ({100*x['done']/x['total']:.0f}%)" if isinstance(x["total"], int) and x["total"] else ""
        eta = f"ETA {x['eta_h']:.1f} h" if x["eta_h"] is not None else "ETA ?"
        mu = f"{x['mean_unit_s']:.0f} s/unidad" if x["mean_unit_s"] else "- s/unidad"
        L.append(f"[{s}] {x['done']}/{tot}{pct} · ok={x['ok']} skip={x['skipped']} err={x['errors']} · {mu} · {eta}"
                 f"{' · TERMINADA' if x['finished'] else ''}")
        L.append(f"        acumulado: {x['cumulative_end']} terminadas, {x['cumulative_errors']} errores"
                 + (f" · en curso: {', '.join(map(str, x['running']))}" if x["running"] and not x["finished"] else ""))
        if "gate_n" in x:
            L.append(f"        compuerta: {x['gate_passed']}/{x['gate_n']}"
                     + (f" ({x['gate_rate']:.0%})" if "gate_rate" in x else "")
                     + (f" · mediana val_pr_auc {x['val_pr_auc_median']}" if "val_pr_auc_median" in x else ""))
        if x.get("no_aplica"):
            L.append("        no_aplica: " + ", ".join(f"{k} {v}" for k, v in x["no_aplica"].items()))
        for e in x["last_errors"][-3:]:
            L.append(f"        ERROR {e['run_id']}: {e['msg']}")
    for s, h in r["heartbeats"].items():
        L.append(f"  latido {s}: hace {h['age_min']} min · {h['run_id']} {h['note'] or ''}")
    L.append("-- señales de alerta --")
    if not r["alerts"]:
        L.append("  ninguna")
    for al in r["alerts"]:
        L.append(f"  [{'CRÍTICA' if al['level'] == 'critical' else 'aviso'}] {al['msg']}")
    return "\n".join(L)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--once", action="store_true", help="un informe y sale (default)")
    g.add_argument("--watch", type=float, metavar="N", help="refresca cada N minutos")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--log-dir", default="./runs_v4")
    ap.add_argument("--stale-min", type=float, default=45)
    ap.add_argument("--stale-warn-min", type=float, default=15)
    ap.add_argument("--oom-streak", type=int, default=2)
    ap.add_argument("--gate-min-n", type=int, default=10)
    ap.add_argument("--gate-min-rate", type=float, default=0.30)
    ap.add_argument("--prauc-lo", type=float, default=0.20)
    ap.add_argument("--prauc-hi", type=float, default=0.95)
    ap.add_argument("--prauc-min-n", type=int, default=3, help="mínimo de modelos para juzgar la mediana")
    ap.add_argument("--noapl-max", type=float, default=0.50)
    ap.add_argument("--noapl-min-n", type=int, default=2)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    while True:
        r = build(a)
        if a.json:
            print(json.dumps(r, ensure_ascii=False, default=str))
        else:
            if a.watch:
                os.system("clear" if os.name != "nt" else "cls")
            print(render(r))
        sys.stdout.flush()
        if not a.watch:
            return r["exit_code"]
        time.sleep(a.watch * 60)


if __name__ == "__main__":
    sys.exit(main())
