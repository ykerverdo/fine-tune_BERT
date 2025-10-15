from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator
import pytorch_lightning as pl

class SquadDataModule(pl.LightningDataModule):
    def __init__(self, model_name="bert-base-uncased", batch_size=8, max_length=128, stride=32):
        super().__init__()
        # Model and training parameters
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride = stride

    def setup(self, stage=None):
        # Load the SQuAD dataset
        self.dataset = load_dataset("squad")
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # Function to preprocess training examples
        def preprocess_training_examples(examples):
            questions = [q.strip() for q in examples["question"]]
            inputs = self.tokenizer(
                questions,
                examples["context"],
                max_length=self.max_length,
                truncation="only_second",
                stride=self.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            answers = examples["answers"]
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_map[i]
                answer = answers[sample_idx]
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label is (0, 0)
                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            return inputs

        small_train_dataset = self.dataset["train"].shuffle(seed=42).select(range(1000))
        # Apply preprocessing to the training dataset
        self.train_dataset = small_train_dataset.map(
            preprocess_training_examples,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )

    # Create a DataLoader for training
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=default_data_collator)