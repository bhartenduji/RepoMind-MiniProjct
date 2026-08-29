import torch
from torch.utils.data import Dataset


class BugPatchDataset(Dataset):

    def __init__(
        self,
        records,
        tokenizer,
        max_length=256,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

        vocab = tokenizer.get_vocab()

        if "<PAD>" not in vocab:
            raise ValueError(
                "Tokenizer vocabulary does not contain <PAD>."
            )

        if "<BOS>" not in vocab:
            raise ValueError(
                "Tokenizer vocabulary does not contain <BOS>."
            )

        if "<EOS>" not in vocab:
            raise ValueError(
                "Tokenizer vocabulary does not contain <EOS>."
            )

        self.pad_token_id = vocab["<PAD>"]
        self.bos_token_id = vocab["<BOS>"]
        self.eos_token_id = vocab["<EOS>"]

        if not isinstance(records, list):
            raise TypeError(
                "records must be a list of dictionaries."
            )

        for index, record in enumerate(records):

            if not isinstance(record, dict):
                raise TypeError(
                    f"Record {index} is not a dictionary."
                )

            if "label" not in record:
                raise KeyError(
                    f"Record {index} does not contain 'label'."
                )

            if record["label"] not in (0, 1):
                raise ValueError(
                    f"Record {index} has invalid label: "
                    f"{record['label']}. Expected 0 or 1."
                )

    def __len__(self):
        return len(self.records)

    def encode_patch(self, patch):

        if patch is None:
            patch = ""

        if not isinstance(patch, str):
            patch = str(patch)

        token_ids = self.tokenizer.encode(
            patch
        ).ids

        token_ids = (
            [self.bos_token_id]
            + token_ids
            + [self.eos_token_id]
        )

        if len(token_ids) > self.max_length:

            content_length = (
            self.max_length - 2
        )

            head_length = (
            content_length // 2
        )

            tail_length = (
            content_length - head_length
        )

            content = token_ids[1:-1]

            token_ids = (
                [self.bos_token_id]
                + content[:head_length]
                + content[-tail_length:]
                + [self.eos_token_id]
        )

            attention_mask = [
        1
        ] * len(token_ids)

            padding_length = (
                self.max_length
                - len(token_ids)
        )

        if padding_length > 0:

            token_ids.extend(
                [self.pad_token_id]
                * padding_length
            )

            attention_mask.extend(
                [0]
                * padding_length
            )

        return (
            token_ids,
            attention_mask,
        )

    def __getitem__(self, index):

        record = self.records[index]

        input_ids, attention_mask = (
            self.encode_patch(
                record.get(
                    "patch",
                    ""
                )
            )
        )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),

            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),

            "label": torch.tensor(
                record["label"],
                dtype=torch.long,
            ),
        }


if __name__ == "__main__":

    print(
        "BugPatchDataset module loaded successfully."
    )
