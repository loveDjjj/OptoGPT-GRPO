from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class SpectralStructureTokenizer:
    tokens: list[str]

    def __post_init__(self) -> None:
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def encode(self, structure_tokens: list[str]) -> list[int]:
        body = [self.token_to_id.get(token, self.unk_token_id) for token in structure_tokens]
        return [self.bos_token_id, *body, self.eos_token_id]

    def decode(self, token_ids: list[int]) -> list[str]:
        decoded: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token[int(token_id)]
            if token in {self.pad_token, self.bos_token}:
                continue
            if token == self.eos_token:
                break
            decoded.append(token)
        return decoded

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "vocab.json").write_text(
            json.dumps(self.tokens, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        (output_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "pad_token": self.pad_token,
                    "bos_token": self.bos_token,
                    "eos_token": self.eos_token,
                    "unk_token": self.unk_token,
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, output_dir: str | Path) -> "SpectralStructureTokenizer":
        output_dir = Path(output_dir)
        tokens = json.loads((output_dir / "vocab.json").read_text(encoding="utf-8"))
        return cls(tokens=list(tokens))
