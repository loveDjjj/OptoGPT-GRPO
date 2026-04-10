from __future__ import annotations

from dataclasses import dataclass


SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]


@dataclass(frozen=True)
class TokenVocabulary:
    special_tokens: list[str]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]


def build_token_vocab(material_names: list[str], thickness_values_nm: list[int]) -> TokenVocabulary:
    tokens = list(SPECIAL_TOKENS)
    for material in sorted(material_names):
        for thickness_nm in thickness_values_nm:
            tokens.append(f"{material}_{thickness_nm}")
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return TokenVocabulary(
        special_tokens=list(SPECIAL_TOKENS),
        token_to_id=token_to_id,
        id_to_token=id_to_token,
    )
