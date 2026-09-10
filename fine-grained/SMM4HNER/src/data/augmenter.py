"""Data augmentation using an LLM via Ollama and a local Contextual MLM.

Strategy: negate (default)
    For each sentence that is in first person (singular or plural) **and**
    contains at least one NER tag other than ``'O'``, the LLM rewrites the
    sentence in a different grammatical person and all NER tags are set to
    ``'O'`` in the augmented output.  Sentences that do not meet both
    criteria are skipped.

Strategy: mangle
    For each sentence that contains at least one NER tag other than ``'O'``,
    the LLM rewrites the sentence replacing all named entities (tokens with a
    non-``'O'`` NER tag) with compatible alternatives while keeping the
    sentence in first person (singular or plural).  The response must follow
    the same ``token label ner_tag`` (space-separated) one-per-line format
    used in the prompt so that NER tags are preserved in the augmented output.

Strategy: contextual
    Uses a Masked Language Model (MLM) and Contextual embeddings to swap
    named entities with contextually accurate replacements. It then filters
    the candidates through an NLI model to reject logical contradictions.

Usage (programmatic)::

    from data.augmenter import augment
    augment("train.csv", "train_aug.csv", strategy="negate")
    augment("train.csv", "train_aug.csv", strategy="mangle")
    augment("train.csv", "train_aug.csv", strategy="contextual")
"""

import ast
import copy
import json
import re
import sys
from tqdm import tqdm
import random
from typing import Optional

import pandas as pd
import requests
from data.span_aug import NERSpan, NERSpanAugmenter, NERExample


# Import the ML pipeline logic from our separate file
from data.context_aug import ContextualAugmenter, NLIValidator

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4:31b"
PLACEHOLDER_LABEL = "_"  # Default (background) label used for augmented tokens.

# First-person singular and plural pronouns used for pre-filtering.
FIRST_PERSON_PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
}

# ---------------------------------------------------------------------------
# Internal helpers (LLM stuff remains untouched)
# ---------------------------------------------------------------------------

def _has_first_person(tokens: list) -> bool:
    return any(t.lower() in FIRST_PERSON_PRONOUNS for t in tokens)

def _has_ner(ner_tags: list) -> bool:
    return any(tag != "O" for tag in ner_tags)

def _build_negate_prompt(tokens: list, labels: list, ner_tags: list) -> str:
    rows = "\n".join(f"{tok}\t{lbl}\t{ner}" for tok, lbl, ner in zip(tokens, labels, ner_tags))
    return (
        "You are a data augmentation assistant.\n"
        "Below is a sentence annotated with NER labels, one token per line "
        "(tab-separated: token, label, NER tag):\n\n"
        f"{rows}\n\n"
        "Rules:\n"
        "1. If the sentence is NOT in first-person singular or plural "
        "(I / me / my / mine / myself / we / us / our / ours / ourselves), "
        "output nothing (empty string).\n"
        "2. If the sentence has NO NER tags other than 'O', output nothing.\n"
        "3. Otherwise, rewrite the sentence changing all first-person references "
        "to third-person (he / she / they / them / their / theirs) and output "
        "ONLY the new tokens as a JSON array of strings.\n\n"
        "Output format: One word per line"
        "or an empty string. No explanation."
    )

def _call_ollama(prompt: str, model: str) -> Optional[str]:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=600,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.RequestException as exc:
        print(f"[augmenter] Ollama request failed: {exc}", file=sys.stderr)
        return None

def _parse_negate_tokens(response: str) -> Optional[list]:
    if not response:
        return None
    start = response.find("[")
    end = response.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        tokens = json.loads(response[start: end + 1])
        if isinstance(tokens, list) and tokens:
            return [str(t) for t in tokens]
    except json.JSONDecodeError:
        pass
    return None

def _build_mangle_prompt(tokens: list, labels: list, ner_tags: list) -> str:
    rows = "\n".join(f"{tok} {lbl} {ner}" for tok, lbl, ner in zip(tokens, labels, ner_tags))
    return (
        "You are a data augmentation assistant.\n"
        "Task: Create a new annotated example by replacing every token whose "
        "NER tag is NOT 'O' with a different, realistic, and compatible "
        "alternative of the same entity type. \n"
        "Instruction 1: Keep all tokens whose NER tag is 'O' unchanged.  \n"
        "Instruction 2: Do NOT change the grammatical person — if the original sentence is in first person (I / me / my / we / us / our), keep it in first person. \n"
        "Instruction 3: Only allowed labels are ClinicalImpacts and SocialImpacts. Do not get creative inventing new labels\n"
        "Instruction 4: If the original sentence has no NER tags other than 'O', output nothing (empty string).\n"
        "Instruction 5: Keep the labels consistent with the ner_tags. For instance ClinicalImpacts should be aligned with ner_tags such as B-ClinicalImpacts, I-ClinicalImpacts, etc.  Do not get creative inventing new labels or ner_tags.\n\n"
        "Output format: one token per line, space-separated as "
        "'token label NER_tag'.  Output ONLY the annotated lines.  "
        "No explanation, no markdown, no extra text.\n"
        "Below is the input sentence annotated with NER labels, one token per line "
        "(space-separated: token, label, NER tag):\n\n"
        f"{rows}\n\n"
    )

def _parse_mangle_response(response: str) -> Optional[tuple]:
    if not response:
        return None
    tokens, labels, ner_tags = [], [], []
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            if len(parts) == 2:
                parts = [parts[0], PLACEHOLDER_LABEL, parts[1]]
            else:
                continue
        tokens.append(parts[0])
        labels.append(parts[1])
        ner_tags.append(parts[2])
    if not tokens:
        return None
    return tokens, labels, ner_tags


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _augment_negate(input_file: str, output_file: str, model: str) -> None:
    df = pd.read_csv(input_file)
    augmented_rows = []
    total = len(df['ID'])
    cur = 0
    for _, row in df.iterrows():
        cur += 1
        print(f"{cur}/{total}")
        tokens = ast.literal_eval(row["tokens"])
        labels = ast.literal_eval(row["labels"])
        ner_tags = ast.literal_eval(row["ner_tags"])
        sentence_id = row["ID"]

        if not _has_first_person(tokens) or not _has_ner(ner_tags):
            continue

        prompt = _build_negate_prompt(tokens, labels, ner_tags)
        response = _call_ollama(prompt, model)
        new_tokens = _parse_negate_tokens(response)

        if new_tokens is None:
            continue

        n = len(new_tokens)
        augmented_rows.append({
            "tokens": str(new_tokens),
            "labels": str([PLACEHOLDER_LABEL] * n),
            "ner_tags": str(["O"] * n),
            "ID": f"{sentence_id}_aug",
        })

    out_df = pd.DataFrame(augmented_rows, columns=["tokens", "labels", "ner_tags", "ID"])
    out_df.to_csv(output_file, index=False)
    print(f"[augmenter] Written {len(augmented_rows)} augmented rows to {output_file}")


def _augment_mangle(input_file: str, output_file: str, model: str) -> None:
    df = pd.read_csv(input_file)
    augmented_rows = []
    total = len(df['ID'])
    cur = 0
    for _, row in df.iterrows():
        cur += 1
        print(f"{cur}/{total}")
        tokens = ast.literal_eval(row["tokens"])
        labels = ast.literal_eval(row["labels"])
        ner_tags = ast.literal_eval(row["ner_tags"])
        sentence_id = row["ID"]

        if not _has_ner(ner_tags):
            continue

        prompt = _build_mangle_prompt(tokens, labels, ner_tags)
        response = _call_ollama(prompt, model)
        parsed = _parse_mangle_response(response)

        if parsed is None:
            continue

        new_tokens, new_labels, new_ner_tags = parsed
        augmented_rows.append({
            "tokens": str(new_tokens),
            "labels": str(new_labels),
            "ner_tags": str(new_ner_tags),
            "ID": f"{sentence_id}_aug",
        })

    out_df = pd.DataFrame(augmented_rows, columns=["tokens", "labels", "ner_tags", "ID"])
    out_df.to_csv(output_file, index=False)
    print(f"[augmenter] Written {len(augmented_rows)} augmented rows to {output_file}")


def _augment_contextual(input_file: str, output_file: str, model: str) -> None:
    """
    Contextual strategy: Performs multi-slot augmentation within a single example.
    Swaps multiple entity types and 'O' labels in one pass for higher complexity.
    """
    df = pd.read_csv(input_file)
    augmented_rows = []
    total = len(df['ID'])

    mlm_model = 'roberta-base' if model == DEFAULT_MODEL else model
    nli = NLIValidator(model_name='cross-encoder/nli-deberta-v3-small')
    augmenter = ContextualAugmenter(
        target_label="PLACEHOLDER",
        mlm_model_path=mlm_model,
        nli_validator=nli
    )

    for idx, row in df.iterrows():
        print(f"Contextual Augmenting: {idx + 1}/{total}")
        # Start with a fresh copy of the original tokens
        current_tokens = ast.literal_eval(row["tokens"])
        labels = ast.literal_eval(row["labels"])
        ner_tags = ast.literal_eval(row["ner_tags"])
        sentence_id = row["ID"]

        # 1. Identify all unique label types + the 'O' class
        available_targets = list(set(tag[2:] for tag in ner_tags if tag.startswith("B-")))
        available_targets.append("O")

        # 2. Select which slots to mutate (e.g., up to 3 different types at once)
        num_to_mutate = min(len(available_targets), 3)
        targets_to_mutate = random.sample(available_targets, num_to_mutate)

        # 3. Iteratively augment the SAME token list
        changed_any = False
        for target in targets_to_mutate:
            augmenter.target_label = target
            # Note: We pass current_tokens, which may have already been modified in this loop
            new_tokens, _ = augmenter.augment(current_tokens, ner_tags)

            if new_tokens != current_tokens:
                current_tokens = new_tokens
                changed_any = True

        # 4. Save the single, multi-augmented result
        if changed_any:
            augmented_rows.append({
                "tokens": str(current_tokens),
                "labels": str(labels),
                "ner_tags": str(ner_tags),
                "ID": f"{sentence_id}_multi_aug",
            })

    # Create the augmented dataframe
    aug_df = pd.DataFrame(augmented_rows, columns=["tokens", "labels", "ner_tags", "ID"])

    # 5. Combine Original + Augmented and Shuffle
    final_df = pd.concat([df, aug_df], ignore_index=True)
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    final_df.to_csv(output_file, index=False)
    print(f"[augmenter] Multi-Slot Augmentation Complete.")
    print(f"Original: {len(df)} | Augmented: {len(aug_df)} | Final: {len(final_df)}")


def _reconstruct_sentence_with_offsets(tokens: list[str], ner_tags: list[str]) -> tuple[str, list[NERSpan]]:
    """
    Reconstructs the sentence.
    - NER tags (B/I) are grouped into single spans.
    - 'O' tags are treated as INDIVIDUAL single-token spans.
    """
    if len(tokens) != len(ner_tags):
        raise ValueError(
            f"Token/ner_tags length mismatch: {len(tokens)} tokens vs {len(ner_tags)} ner_tags. "
            "Each token must have exactly one NER tag."
        )
    sentence = ""
    spans = []
    current_ner_span = None

    for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
        start_offset = len(sentence)
        sentence += token
        end_offset = len(sentence)

        if tag.startswith("B-"):
            if current_ner_span: spans.append(current_ner_span)
            current_ner_span = {"text": token, "label": tag[2:], "start": start_offset, "end": end_offset}
        elif tag.startswith("I-") and current_ner_span:
            current_ner_span["text"] += " " + token
            current_ner_span["end"] = end_offset
        else:
            # If we were in an NER span, close it
            if current_ner_span:
                spans.append(current_ner_span)
                current_ner_span = None

            # If it's an 'O', add it as its own unique 1-token span
            if tag == "O":
                spans.append({"text": token, "label": "O", "start": start_offset, "end": end_offset})

        if i < len(tokens) - 1:
            sentence += " "

    if current_ner_span:
        spans.append(current_ner_span)

    return sentence, [NERSpan(**s) for s in spans]


def _spans_to_token_ner(sentence: str, spans: list[NERSpan]) -> tuple[list[str], list[str], list[str]]:
    """
    Convert sentence + char-level spans to (tokens, labels, ner_tags) with 1:1 alignment.

    Each token is matched to the span that contains its character range.
    Multi-token spans get B- for the first token and I- for continuations.
    """
    tokens = []
    token_char_spans = []
    for m in re.finditer(r"\S+", sentence):
        tokens.append(m.group())
        token_char_spans.append((m.start(), m.end()))

    if not tokens:
        return [], [], []

    labels = []
    ner_tags = []
    current_span_idx = -1  # Track which span we're inside for I- tagging

    for i, (tok_start, tok_end) in enumerate(token_char_spans):
        # Find which span contains this token (token overlaps span: tok_start < span.end and tok_end > span.start)
        matched_span = None
        matched_idx = -1
        for j, span in enumerate(spans):
            if tok_start < span.end and tok_end > span.start:
                matched_span = span
                matched_idx = j
                break

        if matched_span is None:
            current_span_idx = -1
            labels.append(PLACEHOLDER_LABEL)
            ner_tags.append("O")
        else:
            label = matched_span.label if matched_span.label != "O" else PLACEHOLDER_LABEL
            labels.append(label)

            if matched_span.label == "O":
                ner_tags.append("O")
                current_span_idx = -1
            else:
                if matched_idx != current_span_idx:
                    ner_tags.append(f"B-{matched_span.label}")
                    current_span_idx = matched_idx
                else:
                    ner_tags.append(f"I-{matched_span.label}")

    return tokens, labels, ner_tags


def _augment_multi_span(input_file, output_file, model, multiplier=1, max_mutations=10):
    """
    Sequentially augments multiple spans to generate `multiplier` new examples per row.
    """
    df = pd.read_csv(input_file)
    augmenter = NERSpanAugmenter(llm_model=model, lazy_load=False)
    aug_rows = []

    pronouns = {
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
        "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "who", "whom", "whose", "which", "that"
    }

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting Spans"):
        try:
            tokens = ast.literal_eval(row["tokens"])
            ner_tags = ast.literal_eval(row["ner_tags"])

            # Reconstruct original text
            orig_text, orig_spans = _reconstruct_sentence_with_offsets(tokens, ner_tags)

            ner_indices = [i for i, s in enumerate(orig_spans) if s.label != "O"]
            o_indices = [
                i for i, s in enumerate(orig_spans)
                if s.label == "O" and s.text.lower().strip() not in pronouns
            ]

            all_mutable_indices = ner_indices + o_indices

            if not all_mutable_indices:
                # No spans to augment - keep the original example
                aug_rows.append({
                    "tokens": row["tokens"],
                    "labels": row["labels"],
                    "ner_tags": row["ner_tags"],
                    "ID": row["ID"],
                })
                continue

            # Track generated sentences to ensure 10 UNIQUE augmentations
            seen_sentences = {orig_text.lower()}

            # Try to generate `multiplier` times.
            # We use a slightly higher loop count to account for potential duplicates or failures
            attempts = 0
            successes = 0

            while successes < multiplier and attempts < (multiplier * 2):
                attempts += 1
                current_example = NERExample(orig_text, copy.deepcopy(orig_spans))

                # Randomize how many spans we mutate in this specific pass (1 to max_mutations)
                current_num_mutations = random.randint(1, min(max_mutations, len(all_mutable_indices)))

                # Force at least 1 NER if available, otherwise use O spans
                if ner_indices:
                    target_indices = [random.choice(ner_indices)]
                else:
                    target_indices = [random.choice(o_indices)]
                remaining_pool = [i for i in all_mutable_indices if i not in target_indices]

                if current_num_mutations > 1 and remaining_pool:
                    target_indices.extend(
                        random.sample(remaining_pool, k=min(current_num_mutations - 1, len(remaining_pool))))

                random.shuffle(target_indices)

                # Sequentially mutate
                for span_idx in target_indices:
                    # Pass a temperature > 0 to the augmenter
                    current_example = augmenter.augment(
                        current_example,
                        span_index=span_idx,
                        num_candidates=20,
                        num_beams=30,
                        temperature=0.7  # <-- CRUCIAL FOR DIVERSITY
                    )

                # Check if this generated sequence is unique
                if current_example.sentence.lower() not in seen_sentences:
                    seen_sentences.add(current_example.sentence.lower())
                    successes += 1

                    new_tokens, new_labels, new_ner_tags = _spans_to_token_ner(
                        current_example.sentence, current_example.spans
                    )
                    aug_rows.append({
                        "tokens": str(new_tokens),
                        "labels": str(new_labels),
                        "ner_tags": str(new_ner_tags),
                        "ID": f"{row['ID']}_aug_{successes}"
                    })

        except Exception as e:
            print(f"Error row {idx}: {e}")

    pd.DataFrame(aug_rows).to_csv(output_file, index=False)
    print(f"\n[augmenter] Multi-Span sequence complete for {len(aug_rows)} rows.")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY = {
    "negate": _augment_negate, # (from your provided code)
    "mangle": _augment_mangle, # (from your provided code)
    "contextual": _augment_contextual, # (from your provided code)
    "span_synonym": _augment_multi_span
}

def augment(
    input_file: str,
    output_file: str,
    strategy: str = "negate",
    model: str = DEFAULT_MODEL,
) -> None:
    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {sorted(STRATEGY_REGISTRY)}"
        )
    STRATEGY_REGISTRY[strategy](input_file, output_file, model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Augment SMM4H NER data.")
    parser.add_argument("--input", default="../../data/new_train_data.csv", help="Input CSV path")
    parser.add_argument("--output", default="../../data/new_train_data_aug_span.csv", help="Output CSV path")
    parser.add_argument("--strategy", default="span_synonym", choices=STRATEGY_REGISTRY.keys())
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")

    args = parser.parse_args()

    augment(
        input_file=args.input,
        output_file=args.output,
        strategy=args.strategy,
        model=args.model
    )
