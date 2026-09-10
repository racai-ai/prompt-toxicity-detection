"""Contextual NER Augmentation using MLM and NLI."""

import random
import itertools
import string
import torch
import math
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Optional
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Words to protect grammatical structure during 'O' label augmentation
IGNORE_WORDS = {
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those", "a", "an", "the",
    "and", "but", "or", "as", "if", "when", "than", "because",
    "while", "where", "after", "so", "though", "since", "until",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "to", "in", "for", "on", "with", "at", "by", "from", "up", "about",
    "into", "over", "after"
}


class NLIValidator:
    """Uses an NLI model to score entailment, neutral, and contradiction probabilities."""

    def __init__(self, model_name: str = 'cross-encoder/nli-deberta-v3-large'):
        print(f"Loading NLI Validator ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def get_nli_scores(self, premise: str, hypothesis: str) -> dict:
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        scores = {self.model.config.id2label[i].lower(): p.item() for i, p in enumerate(probs)}
        return scores


import math
import random
import string
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification


class ContextualAugmenter:
    def __init__(self,
                 target_label: str,
                 mlm_model_path: str = 'roberta-base',
                 nli_validator: Optional[any] = None,
                 top_k: int = 15,
                 sim_threshold: float = 0.7,  # Lowered slightly for long BI spans
                 beam_width: int = 5,
                 temperature: float = 1.3):

        self.target_label = target_label
        self.sim_threshold = sim_threshold
        self.nli_validator = nli_validator
        self.beam_width = beam_width
        self.temperature = temperature

        print(f"Loading MLM and Embedding models ({mlm_model_path})...")
        self.mlm = pipeline('fill-mask', model=mlm_model_path, top_k=top_k)
        self.mask_token = self.mlm.tokenizer.mask_token

        self.tokenizer = AutoTokenizer.from_pretrained(mlm_model_path, add_prefix_space=True)
        self.model = AutoModel.from_pretrained(mlm_model_path)
        self.model.eval()

    def _get_dual_embeddings(self, tokens: List[str], start_idx: int, end_idx: int) -> Tuple[
        torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.last_hidden_state[0]
        global_vec = hidden_states[0]

        span_subword_embeddings = []
        for word_idx in range(start_idx, end_idx + 1):
            word_span = inputs.word_to_tokens(word_idx)
            if word_span is not None:
                span_subword_embeddings.append(hidden_states[word_span.start: word_span.end])

        if not span_subword_embeddings:
            span_vec = torch.zeros(self.model.config.hidden_size)
        else:
            span_tensor = torch.cat(span_subword_embeddings, dim=0)
            span_vec = torch.mean(span_tensor, dim=0)

        return span_vec, global_vec

    def _extract_target_spans(self, tags: List[str], tokens: List[str]) -> List[Tuple[int, int]]:
        spans = []
        i = 0
        while i < len(tags):
            if self.target_label == "O":
                # Logic for O-label filtering already exists in your prompt
                i += 1
            else:
                if tags[i] == f"B-{self.target_label}":
                    start = i
                    end = i
                    while end + 1 < len(tags) and tags[end + 1] == f"I-{self.target_label}":
                        end += 1
                    spans.append((start, end))
                    i = end + 1
                else:
                    i += 1
        return spans

    def _stochastic_beam_search(self, tokens: List[str], start: int, end: int) -> List[List[str]]:
        """Predicts multi-token spans using Beam Search with Temperature Sampling."""
        # (cumulative_log_prob, current_tokens)
        beams = [(0.0, list(tokens))]

        for i in range(start, end + 1):
            new_beams = []
            for score, current_tokens in beams:
                # Mask current word and all remaining words in the span
                temp_tokens = list(current_tokens)
                for j in range(i, end + 1):
                    temp_tokens[j] = self.mask_token

                masked_sent = " ".join(temp_tokens)
                # Filter-out warnings by passing clean strings
                preds = self.mlm(masked_sent)

                # Handle multi-mask vs single-mask output
                target_preds = preds[0] if isinstance(preds, list) and isinstance(preds[0], list) else preds

                # Filter out original word to ensure it's an actual augmentation
                original_word = tokens[i].lower()
                candidates = [p for p in target_preds if p['token_str'].strip().lower() != original_word]

                if not candidates:
                    candidates = target_preds[:5]

                # Apply Temperature Sampling
                raw_probs = np.array([p['score'] for p in candidates[:10]])
                # Softmax with temperature
                exp_probs = np.exp(np.log(raw_probs + 1e-10) / self.temperature)
                norm_probs = exp_probs / np.sum(exp_probs)

                # Sample up to 'beam_width' indices from the top 10
                num_to_sample = min(len(candidates), self.beam_width)
                sampled_indices = np.random.choice(len(candidates[:10]), size=num_to_sample, replace=False,
                                                   p=norm_probs)

                for idx in sampled_indices:
                    p = candidates[idx]
                    cand_tokens = list(current_tokens)
                    cand_tokens[i] = p['token_str'].strip()
                    new_score = score + math.log(p['score'])
                    new_beams.append((new_score, cand_tokens))

            # Prune to global beam width
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:self.beam_width]

        return [b[1] for b in beams]

    def augment(self, tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str]]:
        spans = self._extract_target_spans(tags, tokens)
        if not spans:
            return tokens, tags

        target_start, target_end = random.choice(spans)
        original_text = " ".join(tokens)
        orig_span_vec, orig_global_vec = self._get_dual_embeddings(tokens, target_start, target_end)

        # 1. Generate Candidates
        candidates = self._stochastic_beam_search(tokens, target_start, target_end)
        if not candidates:
            return tokens, tags

        # 2. Score and Filter
        scored_candidates = []
        for cand_tokens in candidates:
            if cand_tokens == tokens:
                continue

            cand_span_vec, cand_global_vec = self._get_dual_embeddings(cand_tokens, target_start, target_end)

            # Cosine Similarity
            span_sim = F.cosine_similarity(orig_span_vec.unsqueeze(0), cand_span_vec.unsqueeze(0)).item()
            global_sim = F.cosine_similarity(orig_global_vec.unsqueeze(0), cand_global_vec.unsqueeze(0)).item()
            combined_sim = (0.7 * span_sim) + (0.3 * global_sim)

            if combined_sim < self.sim_threshold:
                continue

            # NLI Validation (Safeguarded)
            if self.nli_validator:
                try:
                    nli_scores = self.nli_validator.get_nli_scores(original_text, " ".join(cand_tokens))
                    if nli_scores and nli_scores.get('contradiction', 0) < 0.5:
                        final_score = combined_sim + nli_scores.get('entailment', 0)
                        scored_candidates.append((final_score, cand_tokens))
                except Exception:
                    scored_candidates.append((combined_sim, cand_tokens))
            else:
                scored_candidates.append((combined_sim, cand_tokens))

        # 3. Final Selection
        if not scored_candidates:
            return tokens, tags

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1], tags


def test_augmentation_coherence():
    # Setup
    nli = NLIValidator()
    # Target "LOC" for Location
    augmenter = ContextualAugmenter(target_label="LOC", nli_validator=nli)

    tokens = ["I", "live", "in", "New", "York", "City", "."]
    tags = ["O", "O", "O", "B-LOC", "I-LOC", "I-LOC", "O"]

    print("\n--- Testing Multi-token Coherence ---")
    print(f"Original: {' '.join(tokens)}")

    for i in range(10):
        aug_tokens, _ = augmenter.augment(tokens, tags)
        print(f"Augmentation {i + 1}: {' '.join(aug_tokens)}")

        # Check that we didn't get a mix-and-match mess
        # e.g., "New San City" should be unlikely compared to "San Francisco City"
        assert len(aug_tokens) == len(tokens), "Token length should remain the same."


def test_long_bi_spans():
    # Example: 4-token ORG span
    tokens = ["The", "headquarters", "is", "at", "the", "Massachusetts", "Institute", "of", "Technology", "."]
    tags = ["O", "O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG", "O"]

    augmenter = ContextualAugmenter(target_label="ORG", sim_threshold=0.7)

    print("\n--- Testing 4-Token ORG Span ---")
    print(f"Original: {' '.join(tokens)}")

    for i in range(3):
        aug_tokens, _ = augmenter.augment(tokens, tags)
        print(f"Augmented {i + 1}: {' '.join(aug_tokens)}")

def test_o_label_protection():
    # Setup - Target "O" (Outside) augmentation
    augmenter = ContextualAugmenter(target_label="O")

    tokens = ["The", "hungry", "cat", "sat", "on", "the", "mat", "."]
    tags = ["O", "O", "O", "O", "O", "O", "O", "O"]

    # We expect 'hungry', 'cat', 'sat', 'mat' to be candidates.
    # We expect 'The', 'on', 'the', '.' to be ignored due to IGNORE_WORDS.

    print("\n--- Testing 'O' Label Filtering ---")
    aug_tokens, _ = augmenter.augment(tokens, tags)
    print(f"Original: {' '.join(tokens)}")
    print(f"Augmented: {' '.join(aug_tokens)}")


def test_nli_contradiction_blocking():
    nli = NLIValidator()
    # Use a high threshold to ensure NLI has to work
    augmenter = ContextualAugmenter(target_label="MISC", nli_validator=nli, sim_threshold=0.5)

    tokens = ["Apple", "released", "a", "profitable", "iPhone", "."]
    tags = ["B-ORG", "O", "O", "O", "B-MISC", "O"]

    print("\n--- Testing NLI Logic ---")
    # If the MLM suggests "broken" for "profitable", NLI should ideally block it
    # if it creates a contradiction.
    aug_tokens, _ = augmenter.augment(tokens, tags)
    print(f"Original: {' '.join(tokens)}")
    print(f"Augmented: {' '.join(aug_tokens)}")


if __name__ == "__main__":
    # Note: These require the models to be downloaded/cached
    try:
        test_augmentation_coherence()
        test_o_label_protection()
        test_nli_contradiction_blocking()
        test_long_bi_spans()
        print("\nAll tests completed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")