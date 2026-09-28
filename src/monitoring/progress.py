"""Registro de progreso de corridas largas.

Cada etapa (entrenar, explicar, analizar) escribe eventos en un JSONL de solo-anexar y
refresca un archivo de latido. ``scripts/monitor_run.py`` lee ambos para mostrar avance,
ETA, errores y detectar corridas colgadas. Nada aquí depende de la GPU.

Formato de cada línea del JSONL::

    {"ts": "...", "host": "...", "stage": "train", "run_id": "...", "event": "end",
     "duration_s": 812.4, "metrics": {...}, "gpu_mem_mb": 3120, "msg": ""}

Eventos: ``plan`` (total esperado de la etapa), ``start``, ``end``, ``skip``, ``error``.
"""
from __future__ import annotations

import json
import os
import socket
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gpu_mem_mb() -> int | None:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated() / 2**20)
    except Exception:
        pass
    return None


class RunLog:
    """JSONL de eventos + latido. Seguro para varias etapas escribiendo al mismo archivo."""

    def __init__(self, log_dir: str | Path, stage: str):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.path = self.dir / "progress.jsonl"
        self.heartbeat = self.dir / f"HEARTBEAT_{stage}"
        self.host = socket.gethostname()

    def _write(self, event: str, run_id: str = "", **fields) -> None:
        rec = {"ts": _now(), "host": self.host, "stage": self.stage, "run_id": run_id,
               "event": event, "gpu_mem_mb": _gpu_mem_mb()}
        rec.update(fields)
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.beat(run_id, event)

    def beat(self, run_id: str = "", note: str = "") -> None:
        tmp = self.heartbeat.with_suffix(".tmp")
        tmp.write_text(json.dumps({"ts": _now(), "run_id": run_id, "note": note}), encoding="utf-8")
        tmp.replace(self.heartbeat)

    def plan(self, total: int, items: list[str] | None = None) -> None:
        self._write("plan", total=total, items=items or [])

    def skip(self, run_id: str, reason: str) -> None:
        self._write("skip", run_id, msg=reason)

    def error(self, run_id: str, msg: str) -> None:
        self._write("error", run_id, msg=msg)

    @contextmanager
    def step(self, run_id: str):
        """Envuelve una unidad de trabajo. Registra start/end/error; relanza la excepción.

        Uso::

            with log.step(run_id) as rec:
                ...
                rec["metrics"] = {"val_pr_auc": 0.41}
        """
        rec: dict = {"metrics": {}}
        t0 = time.monotonic()
        self._write("start", run_id)
        try:
            yield rec
        except Exception as exc:  # noqa: BLE001 - se registra y se relanza
            self._write("error", run_id, duration_s=round(time.monotonic() - t0, 1),
                        msg=f"{type(exc).__name__}: {exc}",
                        trace=traceback.format_exc(limit=5))
            raise
        else:
            self._write("end", run_id, duration_s=round(time.monotonic() - t0, 1),
                        metrics=rec.get("metrics", {}))
