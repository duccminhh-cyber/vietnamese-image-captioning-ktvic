"""A deterministic whitespace tokenizer, preserving segmented underscores."""

from collections import Counter
import json
from pathlib import Path
from typing import Iterable
import unicodedata

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
CAPTION_FIELDS = ("caption", "segment_caption")


def tokenize(text: str) -> list[str]:
    """NFC + lowercase + whitespace split; no segmentation model or download."""
    return unicodedata.normalize("NFC", text).lower().split()


class Vocabulary:
    """Stable token IDs, fit exclusively on the derived training subset."""

    def __init__(self, tokens: list[str], metadata: dict | None = None) -> None:
        if tokens[:4] != list(SPECIAL_TOKENS) or len(tokens) != len(set(tokens)):
            raise ValueError("Vocabulary must start with four unique special tokens")
        self.id_to_token = list(tokens)
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.metadata = metadata or {}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = range(4)

    def __len__(self) -> int:
        return len(self.id_to_token)

    @classmethod
    def build(cls, captions: Iterable[str], min_frequency: int = 2,
              metadata: dict | None = None) -> "Vocabulary":
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")
        counts = Counter(token for caption in captions for token in tokenize(caption))
        # Frequency first; lexical tie-break makes token IDs reproducible.
        tokens = [token for token, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
                  if count >= min_frequency and token not in SPECIAL_TOKENS]
        return cls(list(SPECIAL_TOKENS) + tokens,
                   {**(metadata or {}), "min_frequency": min_frequency,
                    "tokenizer": "unicode_nfc_lower_whitespace_v1"})

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        if max_length is not None and max_length < 2:
            raise ValueError("max_length includes BOS/EOS and must be >= 2")
        tokens = tokenize(text)
        if max_length is not None:
            tokens = tokens[:max_length - 2]
        return [self.bos_id] + [self.token_to_id.get(t, self.unk_id) for t in tokens] + [self.eos_id]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        tokens = []
        for index in ids:
            index = int(index)
            if not 0 <= index < len(self):
                raise ValueError(f"Invalid token ID: {index}")
            token = self.id_to_token[index]
            if token == "<eos>":
                if not skip_special:
                    tokens.append(token)
                break
            # Keep <unk> visible: it is meaningful when inspecting a caption.
            if skip_special and token in ("<pad>", "<bos>"):
                continue
            tokens.append(token)
        return " ".join(tokens)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tokens": self.id_to_token, "metadata": self.metadata},
                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "Vocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["tokens"], data["metadata"])
