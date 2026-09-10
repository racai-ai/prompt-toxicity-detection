"""Dataset analysis utility for SMM4H NER.

Usage (programmatic)::

    from data.analyzer import analyze
    analyze("train.csv", "dev.csv", "report.txt")

Usage (CLI)::

    hner analyze <trainset_file> <devset_file> <report_file>

The report includes:
- Inventory of ClinicalImpacts and SocialImpacts entities per file
- Strict-match overlap/disjoint counts between train and dev
- Fuzzy-match overlap/disjoint counts (bag-of-words with Jaccard similarity)
- List of fuzzy-matched non-identical pairs with match scores and occurrence counts
- Per-category lists of train-only, dev-only, and shared entities with counts
"""

import ast
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

EntityInventory = Dict[str, int]  # entity_text -> total occurrence count


def _extract_entities_from_df(df: pd.DataFrame) -> Tuple[EntityInventory, EntityInventory]:
    """Extract ClinicalImpacts and SocialImpacts entities from a DataFrame.

    Reads ``tokens`` and ``ner_tags`` columns and uses BIO boundaries to
    reconstruct multi-token entity spans.

    Returns
    -------
    clinical : dict[str, int]
        Map from entity text to occurrence count.
    social : dict[str, int]
        Map from entity text to occurrence count.
    """
    clinical: EntityInventory = defaultdict(int)
    social: EntityInventory = defaultdict(int)

    for _, row in df.iterrows():
        tokens = ast.literal_eval(row["tokens"]) if isinstance(row["tokens"], str) else row["tokens"]
        ner_tags = ast.literal_eval(row["ner_tags"]) if isinstance(row["ner_tags"], str) else row["ner_tags"]

        current_tokens: List[str] = []
        current_type: str | None = None

        def _flush():
            if not current_tokens:
                return
            text = " ".join(current_tokens)
            if current_type == "ClinicalImpacts":
                clinical[text] += 1
            elif current_type == "SocialImpacts":
                social[text] += 1

        for token, tag in zip(tokens, ner_tags):
            if tag.startswith("B-"):
                _flush()
                current_tokens = [token]
                current_type = tag[2:]
            elif tag.startswith("I-") and current_tokens:
                current_tokens.append(token)
            else:
                _flush()
                current_tokens = []
                current_type = None

        _flush()

    return dict(clinical), dict(social)


def extract_entities(file_path: str) -> Tuple[EntityInventory, EntityInventory]:
    """Load a CSV and extract entity inventories.

    Parameters
    ----------
    file_path:
        Path to a CSV file with ``tokens`` and ``ner_tags`` columns.

    Returns
    -------
    clinical, social : dict[str, int]
    """
    df = pd.read_csv(file_path)
    return _extract_entities_from_df(df)


# ---------------------------------------------------------------------------
# Strict matching
# ---------------------------------------------------------------------------

def _strict_stats(
    train_inv: EntityInventory, dev_inv: EntityInventory
) -> Tuple[int, int, int, set, set, set]:
    """Compute strict overlap/disjoint statistics.

    Returns
    -------
    n_overlap, n_train_only, n_dev_only : int
        Counts of *unique* entity strings.
    overlap_set, train_only_set, dev_only_set : set
    """
    train_set = set(train_inv.keys())
    dev_set = set(dev_inv.keys())
    overlap = train_set & dev_set
    train_only = train_set - dev_set
    dev_only = dev_set - train_set
    return len(overlap), len(train_only), len(dev_only), overlap, train_only, dev_only


# ---------------------------------------------------------------------------
# Fuzzy matching (bag-of-words Jaccard)
# ---------------------------------------------------------------------------

def _bow(text: str) -> set:
    """Return the bag-of-words (set of tokens) for *text*."""
    return set(text.lower().split())


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between the word sets of *a* and *b*."""
    sa = _bow(a)
    sb = _bow(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


# Minimum Jaccard score to consider two entities a fuzzy match.
FUZZY_THRESHOLD = 0.5


def _fuzzy_match(
    train_inv: EntityInventory, dev_inv: EntityInventory, threshold: float = FUZZY_THRESHOLD
) -> Tuple[
    int, int, int,
    List[Tuple[str, str, float]],
    set, set, set,
]:
    """Fuzzy overlap statistics using Jaccard bag-of-words similarity.

    Identical strings are always considered a match (Jaccard == 1.0).

    Returns
    -------
    n_overlap : int
        Number of train entities that have at least one fuzzy match in dev.
    n_train_only : int
        Number of train entities with no fuzzy match in dev.
    n_dev_only : int
        Number of dev entities with no fuzzy match in train.
    fuzzy_pairs : list of (train_text, dev_text, score)
        Non-identical pairs that were fuzzy-matched (highest-score match kept).
    matched_train : set
    unmatched_train : set
    unmatched_dev : set
    """
    train_entities = list(train_inv.keys())
    dev_entities = list(dev_inv.keys())

    # For each train entity find the best-scoring dev entity above threshold.
    train_matched: set = set()
    dev_matched: set = set()
    fuzzy_pairs: List[Tuple[str, str, float]] = []

    for t in train_entities:
        best_score = 0.0
        best_dev = None
        for d in dev_entities:
            score = _jaccard(t, d)
            if score >= threshold and score > best_score:
                best_score = score
                best_dev = d
        if best_dev is not None:
            train_matched.add(t)
            dev_matched.add(best_dev)
            if t != best_dev:
                fuzzy_pairs.append((t, best_dev, best_score))

    unmatched_train = set(train_entities) - train_matched
    unmatched_dev = set(dev_entities) - dev_matched

    return (
        len(train_matched),
        len(unmatched_train),
        len(unmatched_dev),
        sorted(fuzzy_pairs, key=lambda x: -x[2]),
        train_matched,
        unmatched_train,
        unmatched_dev,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _format_entity_list(
    entities: set, inventory: EntityInventory, label: str
) -> List[str]:
    """Format a sorted list of entity strings with occurrence counts."""
    lines = [f"  [{label}]"]
    for ent in sorted(entities):
        count = inventory.get(ent, 0)
        lines.append(f"    {ent!r}  (x{count})")
    return lines


def _section(title: str) -> str:
    bar = "=" * 60
    return f"\n{bar}\n{title}\n{bar}"


def build_report(
    train_file: str,
    dev_file: str,
    threshold: float = 0.5,
) -> str:
    """Build the full analysis report as a string.

    Parameters
    ----------
    train_file, dev_file:
        Paths to the training and development CSV files.
    threshold:
        Jaccard similarity threshold for fuzzy matching (default: 0.5).

    Returns
    -------
    str : the report text.
    """
    train_clinical, train_social = extract_entities(train_file)
    dev_clinical, dev_social = extract_entities(dev_file)

    lines: List[str] = []

    # ------------------------------------------------------------------ header
    lines.append("SMM4H NER — Dataset Analysis Report")
    lines.append(f"  Train file : {train_file}")
    lines.append(f"  Dev file   : {dev_file}")
    lines.append(f"  Fuzzy threshold (Jaccard) : {threshold:.2f}\n")

    # ---------------------------------------------------------- entity counts
    lines.append(_section("ENTITY INVENTORY"))
    lines.append(f"  Train — ClinicalImpacts : {len(train_clinical)} unique entities "
                 f"({sum(train_clinical.values())} total occurrences)")
    lines.append(f"  Train — SocialImpacts   : {len(train_social)} unique entities "
                 f"({sum(train_social.values())} total occurrences)")
    lines.append(f"  Dev   — ClinicalImpacts : {len(dev_clinical)} unique entities "
                 f"({sum(dev_clinical.values())} total occurrences)")
    lines.append(f"  Dev   — SocialImpacts   : {len(dev_social)} unique entities "
                 f"({sum(dev_social.values())} total occurrences)")

    # ------------------------------------------------------- strict statistics
    lines.append(_section("STRICT MATCH STATISTICS"))

    c_n_ov, c_n_tr, c_n_dev, c_ov, c_tr_only, c_dev_only = _strict_stats(train_clinical, dev_clinical)
    s_n_ov, s_n_tr, s_n_dev, s_ov, s_tr_only, s_dev_only = _strict_stats(train_social, dev_social)

    lines.append("  ClinicalImpacts:")
    lines.append(f"    1. Overlapping entities         : {c_n_ov}")
    lines.append(f"    3. Only in trainset             : {c_n_tr}")
    lines.append(f"    4. Only in devset               : {c_n_dev}")
    lines.append("")
    lines.append("  SocialImpacts:")
    lines.append(f"    2. Overlapping entities         : {s_n_ov}")
    lines.append(f"    5. Only in trainset             : {s_n_tr}")
    lines.append(f"    6. Only in devset               : {s_n_dev}")

    # ------------------------------------------------------- fuzzy statistics
    lines.append(_section("FUZZY MATCH STATISTICS"))

    (
        fc_n_ov, fc_n_tr, fc_n_dev,
        fc_pairs,
        fc_matched, fc_tr_only, fc_dev_only,
    ) = _fuzzy_match(train_clinical, dev_clinical, threshold)

    (
        fs_n_ov, fs_n_tr, fs_n_dev,
        fs_pairs,
        fs_matched, fs_tr_only, fs_dev_only,
    ) = _fuzzy_match(train_social, dev_social, threshold)

    lines.append("  ClinicalImpacts:")
    lines.append(f"    1. Overlapping entities (fuzzy) : {fc_n_ov}")
    lines.append(f"    3. Only in trainset (fuzzy)     : {fc_n_tr}")
    lines.append(f"    4. Only in devset   (fuzzy)     : {fc_n_dev}")
    lines.append("")
    lines.append("  SocialImpacts:")
    lines.append(f"    2. Overlapping entities (fuzzy) : {fs_n_ov}")
    lines.append(f"    5. Only in trainset (fuzzy)     : {fs_n_tr}")
    lines.append(f"    6. Only in devset   (fuzzy)     : {fs_n_dev}")

    # ------------------------------------------- fuzzy non-identical matches
    lines.append(_section("FUZZY NON-IDENTICAL MATCHES"))

    def _format_pairs(pairs, train_inv, dev_inv, category):
        out = [f"  {category}:"]
        if not pairs:
            out.append("    (none)")
            return out
        for train_text, dev_text, score in pairs:
            t_count = train_inv.get(train_text, 0)
            d_count = dev_inv.get(dev_text, 0)
            out.append(
                f"    {train_text!r} -> {dev_text!r}  "
                f"({score:.0%} match)  "
                f"(train x{t_count}, dev x{d_count})"
            )
        return out

    lines.extend(_format_pairs(fc_pairs, train_clinical, dev_clinical, "ClinicalImpacts"))
    lines.append("")
    lines.extend(_format_pairs(fs_pairs, train_social, dev_social, "SocialImpacts"))

    # ------------------------------------------------ per-category entity lists
    lines.append(_section("ENTITY LISTS BY CATEGORY"))

    # --- ClinicalImpacts
    lines.append("\n  -- ClinicalImpacts --")

    lines.append("")
    lines.extend(_format_entity_list(c_ov, {**train_clinical, **dev_clinical}, "Strict matches (train ∩ dev)"))
    lines.append("")
    lines.extend(_format_entity_list(c_tr_only, train_clinical, "Only in trainset (strict)"))
    lines.append("")
    lines.extend(_format_entity_list(c_dev_only, dev_clinical, "Only in devset (strict)"))

    # Fuzzy-matched train entities (non-identical; show counts in both splits)
    fuzzy_only_clinical = [(t, d, sc) for t, d, sc in fc_pairs]
    lines.append("")
    lines.append("  [Fuzzy-matched non-identical pairs — ClinicalImpacts]")
    if fuzzy_only_clinical:
        for train_text, dev_text, score in fuzzy_only_clinical:
            t_count = train_clinical.get(train_text, 0)
            d_count = dev_clinical.get(dev_text, 0)
            lines.append(
                f"    {train_text!r} -> {dev_text!r}  "
                f"({score:.0%})  (train x{t_count}, dev x{d_count})"
            )
    else:
        lines.append("    (none)")

    # --- SocialImpacts
    lines.append("\n  -- SocialImpacts --")

    lines.append("")
    lines.extend(_format_entity_list(s_ov, {**train_social, **dev_social}, "Strict matches (train ∩ dev)"))
    lines.append("")
    lines.extend(_format_entity_list(s_tr_only, train_social, "Only in trainset (strict)"))
    lines.append("")
    lines.extend(_format_entity_list(s_dev_only, dev_social, "Only in devset (strict)"))

    fuzzy_only_social = [(t, d, sc) for t, d, sc in fs_pairs]
    lines.append("")
    lines.append("  [Fuzzy-matched non-identical pairs — SocialImpacts]")
    if fuzzy_only_social:
        for train_text, dev_text, score in fuzzy_only_social:
            t_count = train_social.get(train_text, 0)
            d_count = dev_social.get(dev_text, 0)
            lines.append(
                f"    {train_text!r} -> {dev_text!r}  "
                f"({score:.0%})  (train x{t_count}, dev x{d_count})"
            )
    else:
        lines.append("    (none)")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def analyze(train_file: str, dev_file: str, report_file: str) -> None:
    """Analyze two annotated CSV files and write a text report.

    Parameters
    ----------
    train_file:
        Path to the training CSV file.
    dev_file:
        Path to the dev/validation CSV file.
    report_file:
        Path to the output text report file.
    """
    report = build_report(train_file, dev_file)
    with open(report_file, "w", encoding="utf-8") as fh:
        fh.write(report)
        fh.write("\n")
    print(f"[analyze] Report written to {report_file}")
