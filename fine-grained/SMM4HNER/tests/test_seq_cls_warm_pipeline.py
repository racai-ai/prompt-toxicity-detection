import unittest
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import torch
from models.bert_finetuned import BertFinetunedModel
from models.bert_adapter import BertAdapterModel
from training.trainer import Trainer

HIDDEN_SIZE = 8
ORIGINAL_NUM_LABELS = 3


def get_mock_tokenizer():
    def mock_tokenize_fn(prompts, **kwargs):
        if isinstance(prompts, str):
            n = 1
        else:
            n = len(prompts)
        return {
            "input_ids": [[1, 2, 3]] * n,
            "attention_mask": [[1, 1, 1]] * n
        }
    return MagicMock(side_effect=mock_tokenize_fn)


class TestSeqClsWarmPipeline(unittest.TestCase):
    def setUp(self):
        # Create minimal mock datasets
        self.train_df = pd.DataFrame({
            "prompt": ["text one", "text two"],
            "label": ["safe", "unsafe"],
            "granular_label": ["class_a", "class_b"]
        })
        self.val_df = pd.DataFrame({
            "prompt": ["text three"],
            "label": ["safe"],
            "granular_label": ["class_a"]
        })

    @patch("models.bert_finetuned.AutoTokenizer")
    @patch("models.bert_finetuned.AutoModelForSequenceClassification")
    @patch("models.bert_finetuned.HFTrainer")
    @patch("models.bert_finetuned.TrainingArguments")
    def test_bert_finetuned_warm_start(self, mock_args, mock_trainer, mock_auto_model, mock_tokenizer):
        # Mock the model & tokenizer load
        mock_model_instance = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = get_mock_tokenizer()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Instantiate model with custom model name
        model = BertFinetunedModel(model_name="custom-model")
        self.assertEqual(model.model_name, "custom-model")
        
        # Test training with warm pipeline = True
        model.train_seq_cls(
            train_data=self.train_df,
            val_data=self.val_df,
            output_path="dummy_output",
            max_training_epochs=1,
            warm_pipeline=True
        )
        
        # Check that the first head (label) was loaded from the original custom model
        mock_auto_model.from_pretrained.assert_any_call(
            "custom-model", num_labels=2
        )
        
        # Check that the second head (granular_label) was warm-started from the first head (label_head)
        # using ignore_mismatched_sizes=True
        import os
        expected_warm_start_path = os.path.join("dummy_output", "label_head")
        mock_auto_model.from_pretrained.assert_any_call(
            expected_warm_start_path, num_labels=2, ignore_mismatched_sizes=True
        )

    @patch("models.bert_adapter.AutoTokenizer")
    @patch("adapters.AutoAdapterModel")
    @patch("models.bert_adapter._get_dice_adapter_trainer")
    @patch("models.bert_adapter.TrainingArguments")
    def test_bert_adapter_warm_start(self, mock_args, mock_trainer_getter, mock_auto_adapter, mock_tokenizer):
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model_instance.heads = {}
        mock_model_instance.to.return_value = mock_model_instance
        mock_auto_adapter.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = get_mock_tokenizer()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_trainer_cls = MagicMock()
        mock_trainer_getter.return_value = mock_trainer_cls

        model = BertAdapterModel(model_name="custom-adapter-model")
        self.assertEqual(model.model_name, "custom-adapter-model")
        
        # Simulate that during the second head training (granular), the model has the first head already loaded
        # from load_adapter which we mock
        def load_adapter_side_effect(path, load_as):
            mock_model_instance.heads[load_as] = MagicMock()
            
        mock_model_instance.load_adapter.side_effect = load_adapter_side_effect

        model.train_seq_cls(
            train_data=self.train_df,
            val_data=self.val_df,
            output_path="dummy_output",
            max_training_epochs=1,
            warm_pipeline=True
        )
        
        # Verify first head loads base model from custom-adapter-model
        mock_auto_adapter.from_pretrained.assert_any_call("custom-adapter-model")
        
        # Verify granular head warm starts by loading adapter from label_head path
        import os
        expected_warm_start_path = os.path.join("dummy_output", "label_head")
        mock_model_instance.load_adapter.assert_any_call(expected_warm_start_path, load_as="seq_cls_adapter")
        
        # Verify that delete_head and add_classification_head were called for seq_cls_adapter
        mock_model_instance.delete_head.assert_called_with("seq_cls_adapter")
        mock_model_instance.add_classification_head.assert_called_with("seq_cls_adapter", num_labels=2)

    @patch("training.trainer.detect_dataset_type")
    @patch("training.trainer.save_model_yaml")
    def test_trainer_initializes_with_model_name(self, mock_save_yaml, mock_detect_type):
        mock_detect_type.return_value = "sequence_classification"
        
        # Verify Trainer instantiates model with model_name from config kwargs
        trainer = Trainer(model_type="bert_finetuned")
        
        # We mock __init__ of BertFinetunedModel using autospec=True
        with patch.object(BertFinetunedModel, "__init__", return_value=None, autospec=True) as mock_init:
            # We mock load_data to just test initialization
            with patch("training.trainer.load_data") as mock_load_data:
                mock_load_data.return_value = self.train_df
                try:
                    trainer.train(
                        input_file="dummy_train.csv",
                        validation_file="dummy_val.csv",
                        output_model="dummy_out",
                        model_name="custom-deberta-large"
                    )
                except Exception:
                    # Expect it to fail down the line since we mocked __init__ to return None
                    pass
                # Check that model_name was passed to the constructor
                mock_init.assert_called_once()
                self.assertEqual(mock_init.call_args[1].get("model_name"), "custom-deberta-large")

    @patch("models.bert_finetuned.AutoTokenizer")
    @patch("models.bert_finetuned.AutoModelForTokenClassification")
    def test_bert_finetuned_accepts_file_uri_pretrained_model(self, mock_auto_model, mock_tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_head = torch.nn.Linear(HIDDEN_SIZE, ORIGINAL_NUM_LABELS)
            mock_model_instance = MagicMock()
            mock_model_instance.classifier = original_head
            mock_model_instance.config = MagicMock()
            mock_model_instance._init_weights = MagicMock()
            mock_auto_model.from_pretrained.return_value = mock_model_instance

            model = BertFinetunedModel(model_name=f"file://{tmpdir}")
            model._build_tokenizer_and_model()

            mock_tokenizer.from_pretrained.assert_called_once_with(tmpdir)
            mock_auto_model.from_pretrained.assert_called_once_with(
                tmpdir,
                num_labels=len(model.LABEL_LIST),
                id2label=model.ID2LABEL,
                label2id=model.LABEL2ID,
                ignore_mismatched_sizes=True,
            )
            self.assertIsNot(mock_model_instance.classifier, original_head)
            self.assertEqual(mock_model_instance.classifier.in_features, HIDDEN_SIZE)
            self.assertEqual(mock_model_instance.classifier.out_features, len(model.LABEL_LIST))
            self.assertIsNotNone(mock_model_instance.classifier.bias)
            mock_model_instance._init_weights.assert_called_once()
            self.assertIs(mock_model_instance._init_weights.call_args.args[0], mock_model_instance.classifier)

    @patch("models.bert_adapter.AutoTokenizer")
    @patch("adapters.AutoAdapterModel")
    def test_bert_adapter_accepts_disk_uri_pretrained_model(self, mock_auto_adapter, mock_tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model_instance = MagicMock()
            mock_model_instance.heads = {BertAdapterModel.ADAPTER_NAME: MagicMock()}
            mock_model_instance.to.return_value = mock_model_instance
            mock_auto_adapter.from_pretrained.return_value = mock_model_instance

            model = BertAdapterModel(model_name=f"disk://{tmpdir}")
            model._build_tokenizer_and_model()

            mock_tokenizer.from_pretrained.assert_called_once_with(tmpdir)
            mock_auto_adapter.from_pretrained.assert_called_once_with(tmpdir)
            mock_model_instance.load_adapter.assert_called_once_with(
                tmpdir, load_as=BertAdapterModel.ADAPTER_NAME
            )
            mock_model_instance.delete_head.assert_called_once_with(BertAdapterModel.ADAPTER_NAME)
            mock_model_instance.add_tagging_head.assert_called_once()

    @patch("models.bert_adapter.AutoTokenizer")
    @patch("adapters.AutoAdapterModel")
    @patch("models.bert_adapter._get_dice_adapter_trainer")
    @patch("models.bert_adapter.TrainingArguments")
    def test_bert_adapter_seq_cls_uses_base_model_from_local_adapter_config(
        self, mock_args, mock_trainer_getter, mock_auto_adapter, mock_tokenizer
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(Path(tmpdir) / "adapter_config.json", "w", encoding="utf-8") as fh:
                json.dump({"base_model_name_or_path": "base-model"}, fh)

            mock_model_instance = MagicMock()
            mock_model_instance.heads = {"seq_cls_adapter": MagicMock()}
            mock_model_instance.to.return_value = mock_model_instance
            mock_auto_adapter.from_pretrained.return_value = mock_model_instance
            mock_tokenizer.from_pretrained.return_value = get_mock_tokenizer()

            mock_trainer = MagicMock()
            mock_trainer_getter.return_value = MagicMock(return_value=mock_trainer)

            model = BertAdapterModel(model_name=f"disk://{tmpdir}")
            model._train_single_seq_cls_head(
                train_data=self.train_df[["prompt", "label"]],
                val_data=self.val_df[["prompt", "label"]],
                output_path="dummy_output",
                label_col="label",
                classes=sorted(str(v) for v in self.train_df["label"].unique()),
                max_training_epochs=1,
            )

            mock_tokenizer.from_pretrained.assert_called_once_with("base-model")
            mock_auto_adapter.from_pretrained.assert_called_once_with("base-model")
            mock_model_instance.load_adapter.assert_called_once_with(
                tmpdir, load_as="seq_cls_adapter"
            )
            mock_model_instance.delete_head.assert_called_once_with("seq_cls_adapter")
            mock_model_instance.add_classification_head.assert_called_once()
            mock_trainer.train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
