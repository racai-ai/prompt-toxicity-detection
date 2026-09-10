"""Annotation guidelines (constant)."""

ANNOTATION_GUIDELINES = """\
General Instructions
- Annotations capture only meaningful, self-reported consequences of opioid misuse \
expressed by the individual.
- Both SocialImpacts (e.g., job loss, family disruption) and ClinicalImpacts \
(e.g., withdrawal, depression, hospitalization) are included.
- All annotations must reflect the individual's own lived experience.

Inclusion Criteria
Entities are annotated when they meet ALL of the following:
1. First-person account: A social or clinical impact is annotated only if described \
in a first-person account directly related to the poster.
   Example: "I lost my job" -> annotated; "My brother lost his job" -> NOT annotated.
2. Ambiguous context (assumed impact): When opioid involvement cannot be ruled out, \
the impact is assumed to be related.
   Example: "It caused me to fight with my family" -> "fight with my family" = SocialImpacts.
3. Polysubstance mention: If a post mentions multiple substances, annotate assuming \
opioid misuse contributed.
   Example: "I abuse alcohol and heroin, which has affected my health" -> ClinicalImpacts.
4. Mental health symptoms: Mental health issues are annotated as ClinicalImpacts \
unless explicitly attributed to another cause.
   Included: "I feel depressed all the time."
   Excluded: "We broke up, so I am sad." (clearly linked to breakup, not opioid use)
5. Care-seeking behavior: Mentions of rehab, counseling, or treatment are annotated \
as ClinicalImpacts.
   Example: "I went to rehab last month" -> "went to rehab" is annotated.

Exclusion Criteria
The following are explicitly excluded:
1. Third-person accounts: Impacts involving friends, family, or others.
   Example: "My brother lost his job" -> NOT annotated.
2. Drug names: Mentions of specific drugs are NOT annotated as impacts.
3. Personal pronouns in spans: Personal pronouns (I, he, she, they, my, our) are \
excluded from the annotated span if they do not contribute directly to the impact.
   Example: "I lost my job" -> span: "lost my job" (not "I lost my job").
4. Modifiers in spans: Temporal references, adjectives, adverbs are excluded unless \
integral to meaning.
   Example: "I am feeling really tired and crummy" -> span: "tired and crummy".

BIO Tagging Scheme
- B-ClinicalImpacts: Beginning of a ClinicalImpacts entity span.
- I-ClinicalImpacts: Inside/continuation of a ClinicalImpacts entity span.
- B-SocialImpacts: Beginning of a SocialImpacts entity span.
- I-SocialImpacts: Inside/continuation of a SocialImpacts entity span.
- O: Outside any entity span.
"""
