"""
Medical Entity Normalization Service.

This module provides entity normalization capabilities using the UMLS
(Unified Medical Language System) API to link extracted clinical entities
to standardized medical terminologies (SNOMED-CT, ICD-10).

Architecture Context:
    Entity normalization is an optional post-processing step in the MedAI
    extraction pipeline. After NER models identify entity spans, this service
    can enrich diagnosis entities (DX) with standardized codes.

    The normalization workflow:

    1. Search UMLS for candidate concepts matching entity text
    2. Retrieve atoms (codes) for each candidate CUI
    3. Score candidates using semantic similarity
    4. Select best matching code based on terminology priority

Supported Terminologies:
    - **SNOMED-CT** (SNOMEDCT_US): Preferred for clinical concepts
    - **ICD-10-CM** (ICD10CM): Required for billing and reporting

Similarity Scoring:
    The service uses a hybrid similarity approach combining:

    - Multilingual semantic embeddings (via :mod:`app.services.semantic_sim`)
    - Fuzzy string matching (token sort ratio, WRatio)
    - Jaccard similarity on content tokens
    - Terminology-specific bonuses (SAB priority, preferred terms)

Configuration:
    - ``UMLS_APIKEY``: Required environment variable for UMLS API access
    - Minimum link score threshold: 0.60 (configurable via :class:`NormOptions`)

Usage:
    >>> from app.services.normalizer import normalize_entities, NormOptions
    >>> entities = [{"type": "DX", "text": "neumonía"}]
    >>> opts = NormOptions(enabled=True, systems=["SNOMEDCT_US"])
    >>> normalized = normalize_entities(entities, opts)
    >>> print(normalized[0]["codes"][0]["system"])
    'SNOMEDCT_US'

See Also:
    - :mod:`app.services.pipeline` for integration with extraction
    - :mod:`app.services.semantic_sim` for embedding-based similarity
    - :mod:`app.services.translator` for Spanish-English translation
"""

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

# UMLS API configuration
UMLS_APIKEY = os.getenv("UMLS_APIKEY")
"""UMLS API key from environment. Required for normalization."""

UMLS_BASE = "https://uts-ws.nlm.nih.gov"
"""UMLS REST API base URL."""

UMLS_VER = "current"
"""UMLS version identifier (uses current release)."""

# Terminology mapping by entity type
TYPE_TO_SABS: Dict[str, List[str]] = {
    "DX": ["SNOMEDCT_US", "ICD10CM"],
}
"""
Mapping of entity types to preferred terminology sources (SABs).

Currently only diagnosis entities (DX) are normalized. The order
determines priority when multiple codes are available.
"""

# Term type ranking for preferred term selection
TTY_RANK = {"PT": 3, "FN": 2, "SY": 1}
"""
Term type (TTY) ranking for atom selection.

- PT (Preferred Term): Highest priority
- FN (Full Name): Medium priority
- SY (Synonym): Lowest priority
"""


def _strip_accents(s: str) -> str:
    """
    Remove diacritical marks (accents) from text.

    Args:
        s: Input string with potential accents.

    Returns:
        String with accents removed.

    Example:
        >>> _strip_accents("neumonía")
        'neumonia'
    """
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s or "")
        if unicodedata.category(c) != "Mn"
    )


def _norm(s: str) -> str:
    """
    Normalize string for comparison.

    Applies accent removal, lowercase conversion, and whitespace normalization.

    Args:
        s: Input string.

    Returns:
        Normalized string.
    """
    s = _strip_accents(s).lower().strip()
    return re.sub(r"\s+", " ", s)


def _jaccard(a: str, b: str) -> float:
    """
    Calculate Jaccard similarity between two strings.

    Computes the ratio of shared words to total unique words.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Jaccard similarity score (0.0 to 1.0).
    """
    A, B = set(_norm(a).split()), set(_norm(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


# English stopwords for content token extraction
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
"""Common English stopwords excluded from content token comparison."""


def _content_tokens(s: str) -> set:
    """
    Extract meaningful content tokens from text.

    Removes stopwords and returns unique alphabetic tokens.

    Args:
        s: Input string.

    Returns:
        Set of content tokens.
    """
    toks = re.findall(r"[a-zA-Z]+", _norm(s))
    return {t for t in toks if t not in STOP_EN}


def _string_sim_bilingual(span_es: str, name_en: str) -> float:
    """
    Calculate bilingual similarity between Spanish and English text.

    Uses translation and multiple similarity metrics to compare
    a Spanish entity span with an English concept name.

    Args:
        span_es: Spanish text (entity span).
        name_en: English text (UMLS concept name).

    Returns:
        Similarity score (0.0 to 1.0).

    Algorithm:
        1. Translate Spanish to English
        2. Compute fuzzy string similarity (WRatio, token_sort_ratio)
        3. Compute Jaccard similarity
        4. Apply penalty for extra tokens in target
        5. Combine with weighted average
    """
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

    # Penalty for extra tokens in candidate
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
    Calculate semantic similarity using multilingual embeddings.

    Combines embedding-based similarity with fuzzy string matching
    for robust cross-lingual comparison.

    Args:
        span_es: Spanish text (entity span).
        name_en: English text (UMLS concept name).

    Returns:
        Similarity score (0.0 to 1.0).

    Algorithm:
        1. Compute multilingual embedding similarity
        2. Compute fuzzy string similarity
        3. Compute Jaccard similarity
        4. Apply length ratio penalty
        5. Combine with weighted average (60% embedding, 30% fuzzy, 10% Jaccard)
    """
    if not span_es or not name_en:
        return 0.0

    span_n = _norm(span_es)
    name_n = _norm(name_en)

    s_emb = sim_texts(span_n, name_n)
    s_wr = fuzz.WRatio(span_n, name_n) / 100.0
    s_ts = fuzz.token_sort_ratio(span_n, name_n) / 100.0
    s_jac = _jaccard(span_n, name_n)

    base = max(s_wr, s_ts)
    hybrid = 0.6 * s_emb + 0.3 * base + 0.1 * s_jac

    # Length ratio penalty
    la, lb = max(1, len(span_n)), max(1, len(name_n))
    ratio = min(la, lb) / max(la, lb)
    penalty = 0.0 if ratio >= 0.35 else (0.35 - ratio) * 0.3

    score = max(0.0, min(1.0, hybrid - penalty))
    return score


def _passes_vsac(system: str, code: str, white: Optional[Dict[str, set]]) -> bool:
    """
    Check if a code passes VSAC whitelist filtering.

    Args:
        system: Terminology system (e.g., "SNOMEDCT_US").
        code: Code value.
        white: Optional whitelist mapping system to allowed codes.

    Returns:
        True if code passes filter (or no filter configured).
    """
    if not white:
        return True
    allow = white.get(system)
    return True if not allow else code in allow


def _guess_priority_for_type(ent_type: str, systems: Optional[List[str]]) -> List[str]:
    """
    Determine terminology priority for an entity type.

    Args:
        ent_type: Entity type (e.g., "DX").
        systems: Optional list of allowed systems.

    Returns:
        Ordered list of terminology sources to search.
    """
    base = TYPE_TO_SABS.get(ent_type, [])
    return [s for s in base if (not systems or s in systems)]


# HTTP session with retry configuration
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
    """
    Execute GET request to UMLS API with error handling.

    Args:
        url: API endpoint URL.
        params: Query parameters.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response.

    Raises:
        RuntimeError: If UMLS_APIKEY is not configured.
        requests.HTTPError: On API error responses.
    """
    if not UMLS_APIKEY:
        raise RuntimeError("Missing UMLS_APIKEY in environment (.env)")
    params = dict(params or {})
    params.setdefault("apiKey", UMLS_APIKEY)
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=4096)
def umls_search_cuis(query: str, page_size: int = 25) -> List[Tuple[str, str, float]]:
    """
    Search UMLS for concept identifiers (CUIs) matching a query.

    Performs a word-based search and scores results using semantic similarity.

    Args:
        query: Search query (entity text).
        page_size: Maximum number of results to return.

    Returns:
        List of (CUI, name, similarity_score) tuples, sorted by score descending.

    Note:
        Results are cached to reduce API calls for repeated queries.
    """
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
    """
    Retrieve atoms (terminology codes) for a UMLS concept.

    Args:
        cui: UMLS Concept Unique Identifier.
        sabs: Tuple of source abbreviations to filter (empty = all).
        page_size: Maximum atoms to retrieve.

    Returns:
        List of atom dictionaries with keys: sab, code, name, tty.

    Note:
        Results are cached to reduce API calls.
    """
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
    """
    Select the best atom based on terminology and term type priority.

    Args:
        atoms: List of atom dictionaries.
        target_priority: Ordered list of preferred terminology sources.

    Returns:
        Best matching atom dictionary, or None if no match.

    Algorithm:
        Iterates through terminologies in priority order, selecting
        the atom with the highest term type rank (PT > FN > SY).
    """
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
    """
    Configuration options for entity normalization.

    Attributes:
        enabled: Whether normalization is active.
        min_link_score: Minimum similarity score for code assignment (0.0-1.0).
        max_candidates: Maximum UMLS candidates to evaluate per entity.
        systems: List of allowed terminology systems (None = all configured).
        restrict_types: Entity types to normalize (None = all supported).
        vsac_whitelists: Optional code whitelists by system.

    Example:
        >>> opts = NormOptions(
        ...     enabled=True,
        ...     min_link_score=0.70,
        ...     systems=["SNOMEDCT_US"],
        ...     restrict_types=["DX"]
        ... )
    """

    enabled: bool = True
    min_link_score: float = 0.60
    max_candidates: int = 25
    systems: List[str] | None = None
    restrict_types: Optional[List[str]] = None
    vsac_whitelists: Dict[str, set] | None = None


def normalize_entities(entities: List[Dict], opts: NormOptions) -> List[Dict]:
    """
    Normalize a list of entities with UMLS codes.

    Processes each entity through the UMLS lookup pipeline, assigning
    standardized codes from configured terminology systems.

    Args:
        entities: List of entity dictionaries from extraction.
        opts: Normalization configuration options.

    Returns:
        List of entities with added ``codes`` and ``code`` fields.

    Algorithm:
        For each DX entity:

        1. Search UMLS for candidate concepts
        2. Filter candidates by minimum similarity score
        3. Retrieve atoms for each candidate CUI
        4. Select best atom per terminology priority
        5. Score final candidates with bonuses
        6. Deduplicate and sort codes by score

    Note:
        Only DX (diagnosis) entities are currently normalized.
        Other entity types are returned unchanged.

    Example:
        >>> entities = [{"type": "DX", "text": "neumonía", "start": 0, "end": 8}]
        >>> opts = NormOptions(enabled=True)
        >>> result = normalize_entities(entities, opts)
        >>> print(result[0]["codes"][0]["system"])
        'SNOMEDCT_US'
    """
    if not opts.enabled:
        return entities

    normalized = []
    for e in entities:
        ent_type = e.get("type", "")

        # Only normalize DX entities
        if ent_type != "DX":
            normalized.append(e)
            continue

        # Check type restrictions
        if opts.restrict_types and ent_type not in opts.restrict_types:
            normalized.append(e)
            continue

        # Get terminology priority for entity type
        target_priority = _guess_priority_for_type(ent_type, opts.systems)
        if not target_priority:
            normalized.append(e)
            continue

        # Search UMLS for candidates
        span = e.get("text") or ""
        candidates = umls_search_cuis(span, page_size=opts.max_candidates)
        candidates = [
            (cui, name, sim)
            for cui, name, sim in candidates
            if sim >= opts.min_link_score
        ]

        codes: List[Dict] = []
        for cui, name, base_sim in candidates:
            # Get atoms for candidate CUI
            atoms = umls_atoms_for_cui(cui, sabs=tuple(target_priority))
            if not atoms:
                continue
            best = pick_best_atom(atoms, target_priority)
            if not best:
                continue

            # Check VSAC whitelist
            system, code, disp = best["sab"], best["code"], best.get("name")
            if not _passes_vsac(system, code, opts.vsac_whitelists):
                continue

            # Calculate final score with bonuses
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

        # Deduplicate and sort codes
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
