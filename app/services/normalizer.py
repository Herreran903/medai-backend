from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import requests
from rapidfuzz import fuzz
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.services.semantic_sim import sim_texts
from app.services.translator import translate_es_to_en

UMLS_APIKEY = os.getenv("UMLS_APIKEY")
UMLS_BASE = "https://uts-ws.nlm.nih.gov"
UMLS_VER = "current"

TYPE_TO_SABS: Dict[str, List[str]] = {
    "DX": ["SNOMEDCT_US", "ICD10CM"],
}

TTY_RANK = {"PT": 3, "FN": 2, "SY": 1}


def _strip_accents(s: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s or "")
        if unicodedata.category(c) != "Mn"
    )


def _norm(s: str) -> str:
    s = _strip_accents(s).lower().strip()
    return re.sub(r"\s+", " ", s)


def _jaccard(a: str, b: str) -> float:
    A, B = set(_norm(a).split()), set(_norm(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


STOP_EN = {
    "of",
    "the",
    "and",
    "in",
    "on",
    "to",
    "for",
    "with",
    "without",
    "by",
    "an",
    "a",
}


def _content_tokens(s: str) -> set:
    toks = re.findall(r"[a-zA-Z]+", _norm(s))
    return {t for t in toks if t not in STOP_EN}


def _string_sim_bilingual(span_es: str, name_en: str) -> float:
    """Similitud más conservadora con fuzzy + jaccard para reducir falsos 1.0."""
    if not span_es or not name_en:
        return 0.0
    try:
        span_en = translate_es_to_en(span_es) or span_es
    except Exception:
        span_en = span_es

    span_en_n = _norm(span_en)
    name_en_n = _norm(name_en)

    s_wr = fuzz.WRatio(span_en_n, name_en_n) / 100.0
    s_ts = fuzz.token_sort_ratio(span_en_n, name_en_n) / 100.0
    base = max(s_wr, s_ts)

    s_jac = _jaccard(span_en_n, name_en_n)

    q = _content_tokens(span_en_n)
    c = _content_tokens(name_en_n)
    if q:
        extra = len(c - q)
        penalty = min(0.35, extra * 0.06)
    else:
        penalty = 0.0

    hybrid = 0.7 * base + 0.3 * s_jac
    score = hybrid - penalty
    return max(0.0, min(1.0, score))


def _string_sim_semantic(span_es: str, name_en: str) -> float:
    """
    Similaridad híbrida sin traducción:
    - Embeddings multilingües (ES ↔ EN)
    - Fuzzy estricto (WRatio / token_sort_ratio)
    - Jaccard como ancla de solapamiento
    """
    if not span_es or not name_en:
        return 0.0

    span_n = _norm(span_es)
    name_n = _norm(name_en)

    s_emb = sim_texts(span_n, name_n)  # 0..1
    s_wr = fuzz.WRatio(span_n, name_n) / 100.0  # 0..1
    s_ts = fuzz.token_sort_ratio(span_n, name_n) / 100.0
    s_jac = _jaccard(span_n, name_n)

    base = max(s_wr, s_ts)
    hybrid = 0.6 * s_emb + 0.3 * base + 0.1 * s_jac

    la, lb = max(1, len(span_n)), max(1, len(name_n))
    ratio = min(la, lb) / max(la, lb)
    penalty = 0.0 if ratio >= 0.35 else (0.35 - ratio) * 0.3

    score = max(0.0, min(1.0, hybrid - penalty))
    return score


def _passes_vsac(system: str, code: str, white: Optional[Dict[str, set]]) -> bool:
    if not white:
        return True
    allow = white.get(system)
    return True if not allow else code in allow


def _guess_priority_for_type(ent_type: str, systems: Optional[List[str]]) -> List[str]:
    base = TYPE_TO_SABS.get(ent_type, [])
    return [s for s in base if (not systems or s in systems)]


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "UMLS-Normalizer/1.0"})

_retry = Retry(
    total=3,
    backoff_factor=0.3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)


def _get(url: str, params: dict, timeout: int = 30) -> dict:
    if not UMLS_APIKEY:
        raise RuntimeError("Falta UMLS_APIKEY en entorno (.env)")
    params = dict(params or {})
    params.setdefault("apiKey", UMLS_APIKEY)
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=4096)
def umls_search_cuis(query: str, page_size: int = 25) -> List[Tuple[str, str, float]]:
    # query_en = translate_es_to_en(query)
    url = f"{UMLS_BASE}/rest/search/{UMLS_VER}"
    js = _get(url, {"string": query, "searchType": "words", "pageSize": page_size})
    results = js.get("result", {}).get("results", []) or []
    out: List[Tuple[str, str, float]] = []
    for it in results:
        ui = it.get("ui")
        name = it.get("name") or ""
        if not ui or not ui.startswith("C"):
            continue
        sim = _string_sim_semantic(query, name)
        out.append((ui, name, sim))
    out.sort(key=lambda x: x[2], reverse=True)
    return out


@lru_cache(maxsize=4096)
def umls_atoms_for_cui(
    cui: str, sabs: Tuple[str, ...] = (), page_size: int = 100
) -> List[Dict]:
    url = f"{UMLS_BASE}/rest/content/{UMLS_VER}/CUI/{cui}/atoms"
    js = _get(url, {"pageSize": page_size})
    items = js.get("result", []) or []
    out = []
    sabs_set = set(sabs or ())
    for it in items:
        sab = it.get("rootSource")
        if sabs_set and sab not in sabs_set:
            continue
        out.append(
            {
                "sab": sab,
                "code": str(it.get("code")),
                "name": it.get("name"),
                "tty": it.get("termType"),
            }
        )
    return out


def pick_best_atom(atoms: List[Dict], target_priority: List[str]) -> Optional[Dict]:
    """Elige el mejor átomo según prioridad de SAB y TTY preferidos."""
    best, best_score = None, -1
    for sab in target_priority:
        for a in atoms:
            if a.get("sab") != sab:
                continue
            score = TTY_RANK.get(a.get("tty") or "", 0)
            if score > best_score:
                best, best_score = a, score
    return best


@dataclass
class NormOptions:
    enabled: bool = True
    min_link_score: float = 0.60
    max_candidates: int = 25
    systems: List[str] | None = None
    restrict_types: Optional[List[str]] = None
    vsac_whitelists: Dict[str, set] | None = None


def normalize_entities(entities: List[Dict], opts: NormOptions) -> List[Dict]:
    if not opts.enabled:
        return entities

    normalized = []
    for e in entities:
        ent_type = e.get("type", "")

        if ent_type != "DX":
            normalized.append(e)
            continue

        # (si mantienes restrict_types activas, este chequeo ya es redundante)
        if opts.restrict_types and ent_type not in opts.restrict_types:
            normalized.append(e)
            continue

        target_priority = _guess_priority_for_type(ent_type, opts.systems)
        if not target_priority:
            normalized.append(e)
            continue

        span = e.get("text") or ""
        candidates = umls_search_cuis(span, page_size=opts.max_candidates)
        candidates = [
            (cui, name, sim)
            for cui, name, sim in candidates
            if sim >= opts.min_link_score
        ]

        codes: List[Dict] = []
        for cui, name, base_sim in candidates:
            atoms = umls_atoms_for_cui(cui, sabs=tuple(target_priority))
            if not atoms:
                continue
            best = pick_best_atom(atoms, target_priority)
            if not best:
                continue

            system, code, disp = best["sab"], best["code"], best.get("name")
            if not _passes_vsac(system, code, opts.vsac_whitelists):
                continue

            sab_bonus = (len(target_priority) - target_priority.index(system)) * 0.02
            tty_bonus = 0.02 if (best.get("tty") == "PT") else 0.0
            sim_to_disp = _string_sim_semantic(span, disp or name or "")
            final = max(base_sim, sim_to_disp) + sab_bonus + tty_bonus

            codes.append(
                {
                    "system": system,
                    "code": str(code),
                    "display": disp or name,
                    "score": round(min(final, 1.0), 4),
                    "source": "UMLS",
                }
            )

        if codes:
            seen = set()
            dedup = []
            for c in sorted(codes, key=lambda x: x["score"], reverse=True):
                key = (c["system"], c["code"])
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(c)
            e = {**e, "codes": dedup, "code": dedup[0]["code"]}

        normalized.append(e)

    return normalized
