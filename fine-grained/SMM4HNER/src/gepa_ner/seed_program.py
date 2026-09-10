"""Seed DSPy program code string for GEPA evolution.

GEPA evolves this program by mutating the code string.  The code must
define a ``program`` variable that is an instance of ``dspy.Module``.

The program augments existing training examples: given (tokens, ner_tags),
it produces a different example with its own correct BIO tags for the
augmented text.
"""

SEED_PROGRAM = r'''
import dspy


class NERAugmentation(dspy.Signature):
    """You are an expert at data augmentation for Named Entity Recognition on
    social media posts related to opioid use and misuse.

    Your task is to take an existing annotated example and produce a DIFFERENT
    example that preserves the same semantic meaning and entity structure, but
    with varied wording. 

    To ensure high-quality alignments, you must FIRST output the sequence of valid 
    BIO tags you plan to use for your new sentence. THEN, build the augmented output 
    token-by-token to match your planned sequence. Do NOT copy tags from the input; 
    annotate the augmented text from scratch according to the rules below.

    Entity types:
      - ClinicalImpacts: self-reported clinical consequences of opioid misuse
        (withdrawal, overdose, rehab, side effects, mental health symptoms,
        hospitalization, detox, counseling, etc.).
      - SocialImpacts: self-reported social consequences of opioid misuse
        (job loss, family conflict, relationship breakup, isolation, legal
        troubles, financial problems, custody loss, housing loss, etc.).

    Augmentation strategies (use one or combine):
      - Paraphrase: Reword the sentence while keeping entities semantically
        equivalent (e.g., "lost my job" -> "got fired from work").
      - Entity substitution: Replace entity spans with compatible alternatives
        of the same type (inspired by "mangle" strategy).
      - Style change: Vary formality or phrasing while preserving first-person
        and entity semantics.

    Annotation rules for the augmented output:
      1. Only annotate first-person, self-reported impacts.
      2. Drug names are never part of an entity span.
      3. Personal pronouns are excluded from spans unless integral.
      4. Output valid BIO tags: O, B-ClinicalImpacts, I-ClinicalImpacts,
         B-SocialImpacts, I-SocialImpacts.

    Input format: one token per line, tab-separated as TOKEN\tNER_TAG.
    Output format: a planned label sequence, followed by TOKEN\tNER_TAG per line.
    """

    annotated_input: str = dspy.InputField(
        desc="Source sentence with BIO tags, one token per line: TOKEN\tNER_TAG"
    )
    planned_bio_tags: str = dspy.OutputField(
        desc=(
            "A space-separated sequence of valid BIO tags (e.g., 'O O B-SocialImpacts "
            "I-SocialImpacts O') that defines the exact structural layout and length "
            "of your upcoming augmented sentence."
        )
    )
    augmented_output: str = dspy.OutputField(
        desc=(
            "Token-level BIO annotation of the AUGMENTED sentence, one line "
            "per token, tab-separated: TOKEN\tNER_TAG. The augmented text "
            "must be different from the input. The NER_TAGs must EXACTLY match "
            "the sequence you just proposed in planned_bio_tags. Allowed tags: "
            "O, B-ClinicalImpacts, I-ClinicalImpacts, B-SocialImpacts, I-SocialImpacts."
        ),
    )


class NERAugmenter(dspy.Module):
    def __init__(self):
        self.augmentor = dspy.ChainOfThought(NERAugmentation)

    def forward(self, annotated_input: str):
        result = self.augmentor(annotated_input=annotated_input)
        return dspy.Prediction(
            planned_bio_tags=result.planned_bio_tags,
            augmented_output=result.augmented_output
        )


program = NERAugmenter()
'''