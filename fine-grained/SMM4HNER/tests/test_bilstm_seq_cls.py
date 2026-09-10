import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

try:
    import torch
    from models.bilstm import _BiLSTMSeqCls, BiLSTMModel
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class BiLSTMSeqClsTests(unittest.TestCase):
    def test_forward_returns_two_heads(self):
        model = _BiLSTMSeqCls(
            vocab_size=20,
            w2v_vocab_size=20,
            token_vocab_size=30,
            emb_dim=8,
            hidden_dim=4,
            num_label_classes=3,
            num_granular_classes=5,
            num_layers=1,
        )
        word_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
        w2v_word_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
        subword_ids = torch.tensor(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[7, 8], [9, 0], [0, 0]],
            ],
            dtype=torch.long,
        )
        lengths = torch.tensor([3, 2], dtype=torch.long)

        label_logits, granular_logits = model(word_ids, w2v_word_ids, subword_ids, lengths)

        self.assertEqual(label_logits.shape, (2, 3))
        self.assertIsNotNone(granular_logits)
        self.assertEqual(granular_logits.shape, (2, 5))

    def test_prompt_tokenizer_handles_empty_prompt(self):
        self.assertEqual(BiLSTMModel._prompt_to_tokens(""), ["<EMPTY>"])
        self.assertEqual(BiLSTMModel._prompt_to_tokens("a b"), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
