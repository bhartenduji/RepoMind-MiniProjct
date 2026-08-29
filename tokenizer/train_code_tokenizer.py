import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


TRAIN_PATH = (
    "data/processed/"
    "bug_classifier_splits/train.json"
)

OUTPUT_DIR = Path(
    "data/processed/code_tokenizer"
)

VOCAB_SIZE = 16000
MIN_FREQUENCY = 2


SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<MASK>",
]


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def extract_training_text(records):
    """
    Use patch text only.

    We intentionally do not use commit messages
    because we want the tokenizer to learn code
    and diff structure rather than label keywords.
    """

    texts = []

    for record in records:

        patch = record.get(
            "patch",
            ""
        )

        if patch is None:
            patch = ""

        patch = patch.strip()

        if not patch:
            continue

        texts.append(
            patch
        )

    return texts


def training_iterator(texts):
    """
    Yield one training sample at a time.
    """

    for text in texts:
        yield text


def train_tokenizer(texts):
    """
    Train a Byte-Level BPE tokenizer.
    """

    tokenizer = Tokenizer(
        BPE(
            unk_token="<UNK>"
        )
    )

    tokenizer.pre_tokenizer = (
        ByteLevel(
            add_prefix_space=False
        )
    )

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        training_iterator(
            texts
        ),
        trainer=trainer,
    )

    tokenizer.decoder = (
        ByteLevelDecoder()
    )

    return tokenizer


def save_tokenizer(
    tokenizer,
    output_dir
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer_path = (
        output_dir
        /
        "tokenizer.json"
    )

    tokenizer.save(
        str(tokenizer_path)
    )

    return tokenizer_path


def test_tokenizer(
    tokenizer
):
    example = """
@@ -10,5 +10,7 @@
 def login(user):
-    return user is not None
+    if user is None:
+        raise ValueError("user required")
+    return True
"""

    encoded = tokenizer.encode(
        example
    )

    print(
        "\nTokenizer Test"
    )

    print(
        "-------------------------"
    )

    print(
        f"Token count: "
        f"{len(encoded.ids)}"
    )

    print(
        "\nFirst 50 token IDs:"
    )

    print(
        encoded.ids[:50]
    )

    print(
        "\nFirst 50 tokens:"
    )

    print(
        encoded.tokens[:50]
    )

    decoded = tokenizer.decode(
        encoded.ids
    )

    print(
        "\nDecoded preview:"
    )

    print(
        decoded[:500]
    )


if __name__ == "__main__":

    print(
        "\nLoading training dataset..."
    )

    records = load_json(
        TRAIN_PATH
    )

    print(
        f"Training records: "
        f"{len(records)}"
    )

    texts = extract_training_text(
        records
    )

    print(
        f"Non-empty patches: "
        f"{len(texts)}"
    )

    print(
        "\nTraining Byte-Level BPE tokenizer..."
    )

    tokenizer = train_tokenizer(
        texts
    )

    tokenizer_path = save_tokenizer(
        tokenizer,
        OUTPUT_DIR
    )

    print(
        "\nTokenizer training complete."
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.get_vocab_size()}"
    )

    print(
        f"Saved to: "
        f"{tokenizer_path}"
    )

    test_tokenizer(
        tokenizer
    )