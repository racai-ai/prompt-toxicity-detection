"""Comparison utility for two annotated CSV outputs.

Usage (programmatic)::

    from data.comparator import compare
    compare("gold.csv", "predicted.csv", "diverging.txt")

Output format for each diverging example::

     - <ID>
    token1    gold_tag1    predicted_tag1
    token2    gold_tag2    predicted_tag2
    ...

"""

import ast

import pandas as pd


def _get_ner_tags(row: "pd.Series") -> list:
    """Return the NER tags from a DataFrame row, accepting either column name."""
    if "ner_tags" in row.index:
        return ast.literal_eval(row["ner_tags"])
    if "predicted_ner_tags" in row.index:
        return ast.literal_eval(row["predicted_ner_tags"])
    raise KeyError("Neither 'ner_tags' nor 'predicted_ner_tags' column found in row.")


def compare(file1: str, file2: str, output_file: str) -> None:
    """Compare two annotated CSV files and write diverging examples to *output_file*.

    Parameters
    ----------
    file1:
        Path to the first CSV (treated as the gold-standard annotations).
    file2:
        Path to the second CSV (treated as the predicted annotations).
    output_file:
        Path to the output text file.  Only examples whose ``ner_tags``
        differ between *file1* and *file2* are written.
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Index both DataFrames by ID for fast lookup.
    df1 = df1.set_index("ID")
    df2 = df2.set_index("ID")

    diverging = []
    for example_id in df1.index:
        if example_id not in df2.index:
            continue

        row1 = df1.loc[example_id]
        row2 = df2.loc[example_id]

        tokens = ast.literal_eval(row1["tokens"])
        gold_tags = _get_ner_tags(row1)
        pred_tags = _get_ner_tags(row2)

        if gold_tags == pred_tags:
            continue

        lines = [f" - {example_id}"]
        for token, gold, pred in zip(tokens, gold_tags, pred_tags):
            lines.append(f"{token}\t{gold}\t{pred}")
        diverging.append("\n".join(lines))

    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(diverging))
        if diverging:
            fh.write("\n")

    print(f"[compare] {len(diverging)} diverging examples written to {output_file}")
