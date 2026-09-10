import json
import argparse
import unittest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
from openai import OpenAI
import torch.nn.functional as F
from utils.device import get_device, set_device

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NERSpan:
    text: str
    label: str
    start: int
    end: int


@dataclass
class NERExample:
    sentence: str
    spans: List[NERSpan]


# ---------------------------------------------------------------------------
# Augmenter Class
# ---------------------------------------------------------------------------

class NERSpanAugmenter:
    """
    Pipeline to augment NER examples by replacing spans with contextually
    and categorically appropriate synonyms using RoBERTa and an LLM.
    """
    def __init__(
        self,
        roberta_model: str = "roberta-large",
        vllm_base_url: str = "http://localhost:8000/v1",
        llm_model: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
        lazy_load: bool = True
    ):
        self.roberta_model_name = roberta_model
        self.vllm_base_url = vllm_base_url
        self.llm_model = llm_model
        self.device = set_device(device) if device is not None else get_device()

        self.tokenizer = None
        self.model = None
        self.llm_client = None

        if not lazy_load:
            self.load_models()

    def load_models(self):
        """Loads models into memory if they haven't been loaded yet."""
        if self.tokenizer is None or self.model is None:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(self.roberta_model_name)
            self.model = RobertaForMaskedLM.from_pretrained(self.roberta_model_name).to(self.device)
            self.model.eval()

        if self.llm_client is None:
            self.llm_client = OpenAI(base_url=self.vllm_base_url, api_key="EMPTY")

    def _generate_candidates(
            self,
            example,
            span,
            num_candidates: int,
            num_beams: int,
            alpha: float = 0.5
    ):
        self.load_models()

        inputs = self.tokenizer(example.sentence, return_offsets_mapping=True, return_tensors="pt").to(self.device)
        tokens, offsets = inputs["input_ids"][0], inputs["offset_mapping"][0]

        mask_indices = [i for i, (s, e) in enumerate(offsets) if s >= span.start and e <= span.end and s != e]
        if not mask_indices: return []

        # --- 1. CAPTURE THE INITIAL HIDDEN STATE ---
        with torch.no_grad():
            original_outputs = self.model.roberta(tokens.unsqueeze(0), output_hidden_states=True)
            target_hidden_rep = original_outputs.hidden_states[-1][0, mask_indices].mean(dim=0)

        # --- 2. VECTORIZED BEAM SEARCH WITH HIDDEN INJECTION ---
        base_ids = tokens.clone()
        for idx in mask_indices: base_ids[idx] = self.tokenizer.mask_token_id
        beams = [(base_ids, 0.0)]

        for m_idx in mask_indices:
            # BATCHING: Stack all current beams into a single tensor [num_beams, seq_len]
            current_b_ids = torch.stack([b[0] for b in beams])

            with torch.no_grad():
                # Single forward pass for all beams
                outputs = self.model.roberta(current_b_ids, output_hidden_states=True)
                last_hidden_state = outputs.last_hidden_state  # (num_beams, seq_len, hidden_size)

                # Broadcast target_hidden_rep to match the batch size
                blended_hidden = (1 - alpha) * last_hidden_state[:, m_idx, :] + (alpha * target_hidden_rep.unsqueeze(0))

                # Pass through LM Head
                logits = self.model.lm_head(blended_hidden)  # (num_beams, vocab_size)

            # Process probabilities across the whole batch
            probs = torch.softmax(logits, dim=-1)
            top_k = num_beams * 3

            # Multinomial sampling works on 2D tensors natively
            sampled_ids = torch.multinomial(probs, num_samples=top_k, replacement=False)  # (num_beams, top_k)

            # Gather log probs efficiently
            log_probs = torch.log(probs.gather(1, sampled_ids) + 1e-10)  # (num_beams, top_k)

            new_beams = []
            for i, (b_ids, b_score) in enumerate(beams):
                for j in range(top_k):
                    tid = sampled_ids[i, j].item()
                    score = log_probs[i, j].item()
                    new_ids = b_ids.clone()
                    new_ids[m_idx] = tid
                    new_beams.append((new_ids, b_score + score))

            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:num_beams]

        # --- 3. FINAL DECODE ---
        scored_candidates = []
        seen = {span.text.lower()}
        for completed_ids, total_log_prob in beams:
            filled_text = self.tokenizer.decode(completed_ids[mask_indices]).strip()
            if filled_text and filled_text.lower() not in seen:
                seen.add(filled_text.lower())
                scored_candidates.append(filled_text)

        return scored_candidates[:num_candidates]

    def _select_best_candidate(
            self,
            example: NERExample,
            span: NERSpan,
            candidates: List[str],
            temperature: float = 0.0
    ) -> Tuple[str, str]:
        """Asks the vLLM model to pick the best synonym via JSON structured output, ensuring perspective is maintained."""
        self.load_models()

        numbered = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(candidates))
        prompt = f"""You are a helpful NLP assistant specializing in Named Entity Recognition (NER).

Original sentence:
  "{example.sentence}"

The span "{span.text}" is labeled as [{span.label}].

Below are candidate synonym replacements for "{span.text}":
{numbered}

Task:
  Select the candidate that best replaces "{span.text}" while adhering to these strict rules:
  1. Preserve semantic meaning and the NER category ({span.label}).
  2. Fit naturally and grammatically into the sentence.
  3. Be lexically different from the original span.
  4. STRICTLY preserve the grammatical person (first, second, or third person). If the original span uses "my", do not select a candidate that uses "his", "their", or "the".

You must respond in valid JSON format containing exactly two keys:
- "reasoning": A brief explanation of why you chose this candidate, confirming it maintains the original grammatical person.
- "selection": The integer number of the best candidate (e.g., 3). If none are suitable, respond with 0.
"""
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=150,
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content.strip()

        try:
            data = json.loads(raw)
            choice = int(data.get("selection", 0))
            reasoning = data.get("reasoning", "No reasoning provided.")

            if 1 <= choice <= len(candidates):
                return candidates[choice - 1], reasoning
        except (json.JSONDecodeError, ValueError) as e:
            reasoning = f"Failed to parse JSON: {e}. Raw output: {raw}"

        fallback_candidate = candidates[0] if candidates else span.text
        return fallback_candidate, reasoning

    def _apply_replacement(self, example: NERExample, span: NERSpan, replacement: str) -> NERExample:
        """Substitutes `replacement` for `span` and recalculates all char offsets."""
        new_sentence = (
            example.sentence[: span.start]
            + replacement
            + example.sentence[span.end :]
        )

        delta = len(replacement) - len(span.text)
        new_spans = []
        for s in example.spans:
            if s is span:
                new_spans.append(NERSpan(
                    text=replacement,
                    label=s.label,
                    start=s.start,
                    end=s.end + delta,
                ))
            elif s.start >= span.end:
                new_spans.append(NERSpan(
                    text=s.text,
                    label=s.label,
                    start=s.start + delta,
                    end=s.end + delta,
                ))
            else:
                new_spans.append(s)

        return NERExample(sentence=new_sentence, spans=new_spans)

    def augment(self, example: NERExample, span_index: int = 0, num_candidates: int = 8, num_beams: int = 3, verbose: bool = False, temperature: float = 0.7) -> NERExample:
        """High-level pipeline method to execute the full augmentation flow."""
        if span_index >= len(example.spans):
            raise IndexError(f"span_index {span_index} out of range for example with {len(example.spans)} spans.")

        span = example.spans[span_index]

        if verbose:
            print(f"Target: '{span.text}' [{span.label}]")

        candidates = self._generate_candidates(example, span, num_candidates, num_beams)
        if not candidates:
            return example

        best_candidate, reasoning = self._select_best_candidate(example, span, candidates, temperature)

        if verbose:
            print(f"Chosen: '{best_candidate}' | Reasoning: {reasoning}")

        return self._apply_replacement(example, span, best_candidate)


# (Assuming the NERSpan, NERExample, and NERSpanAugmenter classes are defined above)

# (Assuming the NERSpan, NERExample, and NERSpanAugmenter classes are defined above)

def run_reddit_demo():
    print("Loading models... (this might take a sec)")
    augmenter = NERSpanAugmenter(
        roberta_model="roberta-large",
        vllm_base_url="http://localhost:8000/v1",  # Ensure vLLM is running!
        llm_model="Qwen/Qwen3-0.6B",
        lazy_load=False
    )

    # Expanded list of SMM4H-HeaRD Task 7 dataset examples
    reddit_posts = [
        # NERExample(
        #     sentence="The brain fog and severe withdrawals made it impossible to function.",
        #     spans=[
        #         NERSpan(text="brain fog", label="ClinicalImpact", start=4, end=13),
        #         NERSpan(text="severe withdrawals", label="ClinicalImpact", start=18, end=36)
        #     ]
        # ),
        # NERExample(
        #     sentence="I ended up losing my job because I kept showing up high.",
        #     spans=[
        #         NERSpan(text="losing my job", label="SocialImpact", start=11, end=24)
        #     ]
        # ),
        # NERExample(
        #     sentence="My drinking completely ruined my marriage last year.",
        #     spans=[
        #         NERSpan(text="ruined my marriage", label="SocialImpact", start=23, end=41)
        #     ]
        # ),
        # NERExample(
        #     sentence="I get terrible night sweats whenever I try to taper off.",
        #     spans=[
        #         NERSpan(text="night sweats", label="ClinicalImpact", start=15, end=27)
        #     ]
        # ),
        NERExample(
            sentence="I Got a DUI over the weekend and now my car is impounded.",
            spans=[
                NERSpan(text="I Got a DUI", label="O", start=0, end=12)
            ]
        ),
        NERExample(
            sentence="The paranoia is the absolute worst part of the comedown.",
            spans=[
                NERSpan(text="paranoia", label="ClinicalImpact", start=4, end=12)
            ]
        )
    ]

    for i, example in enumerate(reddit_posts):
        print(f"\n{'=' * 60}")
        print(f"--- Processing Reddit Post {i + 1} ---")
        print(f"{'=' * 60}")

        # We'll augment the first span (index 0) of each example
        augmented = augmenter.augment(
            example=example,
            span_index=0,
            num_candidates=30,
            num_beams=100,
            verbose=True
        )

        print("\n[RESULT]")
        print(f"Original : {example.sentence}")
        print(f"Augmented: {augmented.sentence}")
        print("New Spans:")
        for s in augmented.spans:
            print(f"  - '{s.text}' [{s.label}] chars [{s.start}:{s.end}]")


if __name__ == "__main__":
    run_reddit_demo()