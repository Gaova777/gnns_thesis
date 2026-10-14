"""Construye bibliografia.bib solo desde Crossref (nada se transcribe a mano).

Entrada: referencias/candidatas.csv con columnas clave,doi,uso.
  - Para cada DOI se consulta https://api.crossref.org/works/<doi>.
  - Se acepta solo type == journal-article (artículos y reviews de revista).
  - Se rechaza si Crossref registra una retractación o corrección de retiro.
  - La entrada @article se genera con autores, título, revista, año, volumen,
    número, páginas y DOI tal como los devuelve Crossref.
Las excepciones (artículo original de un método o dataset, sin DOI de revista) van en
referencias/excepciones.bib, escritas a mano y verificadas en su fuente oficial; este
script solo comprueba que su URL responda.

Salidas: bibliografia.bib (las aceptadas + las excepciones) y referencias/verificacion.csv.
Uso: python scripts/verificar_referencias.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "tesis-xai-gnn/1.0 (mailto:gaova777@utp.edu.co)"}


def crossref(doi: str) -> dict | None:
    url = "https://api.crossref.org/works/" + urllib.request.quote(doi)
    for intento in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)["message"]
        except Exception as e:  # noqa: BLE001
            if "404" in str(e):
                return None
            time.sleep(2 + 3 * intento)
    return None


def url_responde(url: str) -> int:
    try:
        req = urllib.request.Request(url, headers=UA, method="GET")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except Exception as e:  # noqa: BLE001
        m = re.search(r"HTTP Error (\d+)", str(e))
        return int(m.group(1)) if m else -1


def tex(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)          # etiquetas JATS (<i>, <sub>...)
    s = re.sub(r"\s+", " ", s).strip()
    for a, b in [("&amp;", r"\&"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s.replace(r"\\&", r"\&")


def autores(m: dict) -> str:
    out = []
    for a in m.get("author", []):
        if "family" in a:
            out.append(f"{tex(a['family'])}, {tex(a.get('given', ''))}".rstrip(", "))
        elif "name" in a:
            out.append("{" + tex(a["name"]) + "}")
    return " and ".join(out)


def anio(m: dict) -> int | None:
    for k in ("published-print", "issued", "published-online", "published"):
        dp = (m.get(k) or {}).get("date-parts")
        if dp and dp[0] and dp[0][0]:
            return int(dp[0][0])
    return None


def retractado(m: dict) -> bool:
    for u in m.get("update-to", []) + m.get("updated-by", []):
        if "retract" in str(u.get("type", "")).lower() or "withdraw" in str(u.get("type", "")).lower():
            return True
    rel = m.get("relation", {})
    return any("retract" in k.lower() for k in rel)


def entrada(clave: str, m: dict) -> str:
    campos = {
        "author": autores(m),
        "title": "{" + tex(m["title"][0]) + "}",
        "journal": tex((m.get("container-title") or [""])[0]),
        "year": str(anio(m) or ""),
        "volume": m.get("volume", ""),
        "number": m.get("issue", ""),
        "pages": (m.get("page", "") or m.get("article-number", "")).replace("-", "--"),
        "doi": m["DOI"],
    }
    cuerpo = ",\n".join(f"  {k} = {{{v}}}" for k, v in campos.items() if v)
    return f"@article{{{clave},\n{cuerpo}\n}}\n"


def main() -> int:
    filas = list(csv.DictReader(open(RAIZ / "referencias/candidatas.csv", encoding="utf-8")))
    bib, rep = [], []
    for f in filas:
        clave, doi = f["clave"].strip(), f["doi"].strip()
        m = crossref(doi)
        if m is None:
            estado, tipo, titulo, revista, a = "RECHAZADA: DOI no existe en Crossref", "", "", "", ""
        else:
            tipo = m.get("type", "")
            titulo = (m.get("title") or [""])[0]
            revista = (m.get("container-title") or [""])[0]
            a = anio(m)
            if tipo != "journal-article":
                estado = f"RECHAZADA: tipo {tipo}"
            elif retractado(m):
                estado = "RECHAZADA: retractación registrada"
            else:
                estado = "aceptada"
                bib.append(entrada(clave, m))
        rep.append({"clave": clave, "doi": doi, "estado": estado, "tipo": tipo, "anio": a,
                    "revista": revista, "titulo": titulo, "uso": f.get("uso", "")})
        print(f"{estado[:40]:40s} {clave:28s} {a} {revista[:40]}", flush=True)
        time.sleep(0.3)

    exc = (RAIZ / "referencias/excepciones.bib").read_text(encoding="utf-8")
    for clave, url in re.findall(r"@\w+\{([^,]+),.*?url\s*=\s*\{([^}]+)\}", exc, re.S):
        st = url_responde(url)
        rep.append({"clave": clave, "doi": "", "estado": f"excepción (URL {st})", "tipo": "",
                    "anio": "", "revista": "", "titulo": url, "uso": "método o dataset usado"})
        print(f"excepción URL {st:<28} {clave}", flush=True)

    (RAIZ / "bibliografia.bib").write_text(
        "% Generado por scripts/verificar_referencias.py desde Crossref. No editar a mano.\n\n"
        + "\n".join(bib) + "\n% ---- Excepciones (artículo original de un método o dataset) ----\n\n"
        + exc, encoding="utf-8")
    with open(RAIZ / "referencias/verificacion.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rep[0]))
        w.writeheader()
        w.writerows(rep)
    malas = [r for r in rep if not (r["estado"] == "aceptada" or
                                    r["estado"].startswith("excepción (URL 200"))]
    print(f"\n{len(bib)} aceptadas; {len(malas)} con problema")
    return 1 if malas else 0


if __name__ == "__main__":
    sys.exit(main())
