# our_work Spectral Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent `our_work/pretrain` stack that loads the new shard dataset, tokenizes structure sequences, trains a decoder-only spectrum-conditioned causal LM with `Transformers + Trainer`, and saves Hugging Face style checkpoints.

**Architecture:** Treat the `2048`-dimensional `R+T` spectrum as a continuous condition, project it into a fixed number of prefix embeddings, concatenate that prefix with structure token embeddings, and train a causal LM loss only on the structure region. Implement a custom HF model family so the checkpoint, tokenizer, generation, and teacher-forcing scoring interfaces stay clean for later RL work.

**Tech Stack:** Python, PyTorch, Transformers, Datasets, safetensors, PyYAML, pytest

---

## File Structure

### Dataset and tokenizer
- Create: `our_work/pretrain/__init__.py`
- Create: `our_work/pretrain/dataset/__init__.py`
- Create: `our_work/pretrain/dataset/tokenizer.py`
- Create: `our_work/pretrain/dataset/hf_dataset.py`
- Create: `our_work/pretrain/dataset/collator.py`

### Model
- Create: `our_work/pretrain/model/__init__.py`
- Create: `our_work/pretrain/model/configuration_spectral_gpt.py`
- Create: `our_work/pretrain/model/projector.py`
- Create: `our_work/pretrain/model/modeling_spectral_gpt.py`
- Create: `our_work/pretrain/model/generation.py`

### Training scripts and config
- Create: `our_work/pretrain/configs/model/base_gpt.yaml`
- Create: `our_work/pretrain/configs/train/base_train.yaml`
- Create: `our_work/pretrain/trainer/__init__.py`
- Create: `our_work/pretrain/trainer/metrics.py`
- Create: `our_work/pretrain/scripts/run_pretrain.py`
- Create: `our_work/pretrain/scripts/run_eval.py`

### Tests
- Create: `tests/our_work/pretrain/test_tokenizer.py`
- Create: `tests/our_work/pretrain/test_collator.py`
- Create: `tests/our_work/pretrain/test_model_forward.py`
- Create: `tests/our_work/pretrain/test_generation.py`
- Create: `tests/our_work/pretrain/test_training_smoke.py`

### Docs
- Modify: `docs/notes.md`
- Modify: `docs/logs/2026-04.md`

---

### Task 1: Implement the independent structure tokenizer

**Files:**
- Create: `our_work/pretrain/__init__.py`
- Create: `our_work/pretrain/dataset/__init__.py`
- Create: `our_work/pretrain/dataset/tokenizer.py`
- Test: `tests/our_work/pretrain/test_tokenizer.py`

- [ ] **Step 1: Write the failing tests**

```python
from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer


def test_tokenizer_round_trip():
    tokenizer = SpectralStructureTokenizer(
        tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10", "SiO2_20"]
    )
    ids = tokenizer.encode(["Ge_10", "SiO2_20"])
    assert ids == [1, 4, 5, 2]
    assert tokenizer.decode(ids) == ["Ge_10", "SiO2_20"]


def test_tokenizer_maps_unknown_token_to_unk():
    tokenizer = SpectralStructureTokenizer(
        tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10"]
    )
    assert tokenizer.encode(["Missing_30"]) == [1, 3, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/our_work/pretrain/test_tokenizer.py -v`  
Expected: FAIL with `ImportError` for missing tokenizer module

- [ ] **Step 3: Implement the tokenizer**

```python
# our_work/pretrain/dataset/tokenizer.py
from __future__ import annotations

from dataclasses import dataclass


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/our_work/pretrain/test_tokenizer.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/our_work/pretrain/test_tokenizer.py our_work/pretrain/dataset/tokenizer.py
git commit -m "feat: add independent spectral structure tokenizer"
```

---

### Task 2: Implement dataset loading and prefix-aware collation

**Files:**
- Create: `our_work/pretrain/dataset/hf_dataset.py`
- Create: `our_work/pretrain/dataset/collator.py`
- Test: `tests/our_work/pretrain/test_collator.py`

- [ ] **Step 1: Write the failing test**

```python
import torch

from our_work.pretrain.dataset.collator import SpectralCausalCollator
from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer


def test_collator_masks_prefix_positions_with_ignore_index():
    tokenizer = SpectralStructureTokenizer(
        tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10", "SiO2_20"]
    )
    collator = SpectralCausalCollator(tokenizer=tokenizer, prefix_length=3)
    batch = collator(
        [
            {"spectrum_rt": [0.1] * 2048, "structure_tokens": ["Ge_10", "SiO2_20"]},
            {"spectrum_rt": [0.2] * 2048, "structure_tokens": ["Ge_10"]},
        ]
    )
    assert batch["spectra"].shape == (2, 2048)
    assert batch["input_ids"].shape[0] == 2
    assert torch.all(batch["labels"][:, :3] == -100)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/our_work/pretrain/test_collator.py -v`  
Expected: FAIL with `ImportError` for missing collator module

- [ ] **Step 3: Implement the dataset adapter**

```python
# our_work/pretrain/dataset/hf_dataset.py
from __future__ import annotations

from datasets import Dataset, load_dataset


def load_parquet_shards(paths: list[str]) -> Dataset:
    return load_dataset("parquet", data_files=paths, split="train")
```

- [ ] **Step 4: Implement the collator**

```python
# our_work/pretrain/dataset/collator.py
from __future__ import annotations

import torch


class SpectralCausalCollator:
    def __init__(self, tokenizer, prefix_length: int, ignore_index: int = -100) -> None:
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.ignore_index = ignore_index

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        encoded = [self.tokenizer.encode(sample["structure_tokens"]) for sample in samples]
        max_len = max(len(ids) for ids in encoded)
        input_ids = torch.full((len(samples), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        token_attention = torch.zeros((len(samples), max_len), dtype=torch.long)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            token_attention[row, : len(ids)] = 1
        labels = torch.full((len(samples), self.prefix_length + max_len), self.ignore_index, dtype=torch.long)
        labels[:, self.prefix_length :] = input_ids
        labels[:, self.prefix_length] = self.ignore_index
        attention_mask = torch.cat(
            [torch.ones((len(samples), self.prefix_length), dtype=torch.long), token_attention],
            dim=1,
        )
        return {
            "spectra": torch.tensor([sample["spectrum_rt"] for sample in samples], dtype=torch.float32),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/our_work/pretrain/test_collator.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/our_work/pretrain/test_collator.py our_work/pretrain/dataset/hf_dataset.py our_work/pretrain/dataset/collator.py
git commit -m "feat: add spectral dataset adapter and collator"
```

---

### Task 3: Implement the custom config, projector, and forward path

**Files:**
- Create: `our_work/pretrain/model/__init__.py`
- Create: `our_work/pretrain/model/configuration_spectral_gpt.py`
- Create: `our_work/pretrain/model/projector.py`
- Create: `our_work/pretrain/model/modeling_spectral_gpt.py`
- Test: `tests/our_work/pretrain/test_model_forward.py`

- [ ] **Step 1: Write the failing test**

```python
import torch

from our_work.pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM


def test_model_forward_returns_loss_and_logits():
    config = SpectralGPTConfig(
        vocab_size=8,
        spectrum_dim=2048,
        prefix_length=4,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_positions=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = SpectralGPTForCausalLM(config)
    outputs = model(
        spectra=torch.randn(2, 2048),
        input_ids=torch.tensor([[1, 4, 5, 2], [1, 4, 2, 0]], dtype=torch.long),
        attention_mask=torch.tensor([[1] * 8, [1] * 7 + [0]], dtype=torch.long),
        labels=torch.tensor([[-100, -100, -100, -100, -100, 4, 5, 2], [-100, -100, -100, -100, -100, 4, 2, -100]], dtype=torch.long),
    )
    assert outputs.loss is not None
    assert outputs.logits.shape == (2, 8, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/our_work/pretrain/test_model_forward.py -v`  
Expected: FAIL with `ImportError` for missing model classes

- [ ] **Step 3: Implement the config and projector**

```python
# our_work/pretrain/model/configuration_spectral_gpt.py
from __future__ import annotations

from transformers import PretrainedConfig


class SpectralGPTConfig(PretrainedConfig):
    model_type = "spectral_gpt"

    def __init__(
        self,
        vocab_size: int,
        spectrum_dim: int = 2048,
        prefix_length: int = 8,
        n_positions: int = 32,
        n_embd: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.spectrum_dim = spectrum_dim
        self.prefix_length = prefix_length
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
```

```python
# our_work/pretrain/model/projector.py
from __future__ import annotations

import torch
import torch.nn as nn


class SpectrumProjector(nn.Module):
    def __init__(self, spectrum_dim: int, prefix_length: int, hidden_size: int) -> None:
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size
        self.proj = nn.Sequential(
            nn.Linear(spectrum_dim, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, prefix_length * hidden_size),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        prefix = self.proj(spectra)
        return prefix.view(spectra.size(0), self.prefix_length, self.hidden_size)
```

- [ ] **Step 4: Implement the HF model forward**

```python
# our_work/pretrain/model/modeling_spectral_gpt.py
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from our_work.pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from our_work.pretrain.model.projector import SpectrumProjector


class SpectralGPTForCausalLM(PreTrainedModel):
    config_class = SpectralGPTConfig

    def __init__(self, config: SpectralGPTConfig) -> None:
        super().__init__(config)
        backbone_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.n_positions,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            embd_pdrop=config.embd_pdrop,
            resid_pdrop=config.resid_pdrop,
            attn_pdrop=config.attn_pdrop,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            pad_token_id=config.pad_token_id,
        )
        self.backbone = GPT2Model(backbone_config)
        self.projector = SpectrumProjector(config.spectrum_dim, config.prefix_length, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, spectra, input_ids, attention_mask=None, labels=None, **kwargs):
        prefix_embeds = self.projector(spectra)
        token_embeds = self.backbone.wte(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/our_work/pretrain/test_model_forward.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/our_work/pretrain/test_model_forward.py our_work/pretrain/model
git commit -m "feat: add spectral gpt model forward path"
```

---

### Task 4: Add generation and teacher-forcing helpers

**Files:**
- Create: `our_work/pretrain/model/generation.py`
- Test: `tests/our_work/pretrain/test_generation.py`

- [ ] **Step 1: Write the failing test**

```python
import torch

from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer
from our_work.pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from our_work.pretrain.model.generation import generate_structure_tokens
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM


def test_generate_structure_tokens_returns_token_lists():
    tokenizer = SpectralStructureTokenizer(
        tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10", "SiO2_20"]
    )
    config = SpectralGPTConfig(
        vocab_size=len(tokenizer.tokens),
        spectrum_dim=2048,
        prefix_length=2,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_positions=16,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = SpectralGPTForCausalLM(config)
    results = generate_structure_tokens(model, tokenizer, torch.randn(2, 2048), max_new_tokens=3)
    assert len(results) == 2
    assert all(isinstance(item, list) for item in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/our_work/pretrain/test_generation.py -v`  
Expected: FAIL with `ImportError` for missing generation helper

- [ ] **Step 3: Implement a minimal autoregressive generator**

```python
# our_work/pretrain/model/generation.py
from __future__ import annotations

import torch


@torch.inference_mode()
def generate_structure_tokens(model, tokenizer, spectra: torch.Tensor, max_new_tokens: int) -> list[list[str]]:
    batch_size = spectra.size(0)
    input_ids = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=spectra.device)
    for _ in range(max_new_tokens):
        attention_mask = torch.ones(
            (batch_size, model.config.prefix_length + input_ids.size(1)),
            dtype=torch.long,
            device=spectra.device,
        )
        outputs = model(spectra=spectra, input_ids=input_ids, attention_mask=attention_mask)
        next_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_ids], dim=1)
    return [tokenizer.decode(row.tolist()) for row in input_ids]
```

- [ ] **Step 4: Add a teacher-forcing logprob helper**

```python
# our_work/pretrain/model/generation.py
def score_structure_tokens(model, spectra: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(spectra=spectra, input_ids=input_ids, attention_mask=attention_mask)
    log_probs = outputs.logits.log_softmax(dim=-1)
    return log_probs[:, :-1].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/our_work/pretrain/test_generation.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/our_work/pretrain/test_generation.py our_work/pretrain/model/generation.py
git commit -m "feat: add spectral generation and scoring helpers"
```

---

### Task 5: Add config files, Trainer entrypoint, and a smoke training test

**Files:**
- Create: `our_work/pretrain/configs/model/base_gpt.yaml`
- Create: `our_work/pretrain/configs/train/base_train.yaml`
- Create: `our_work/pretrain/trainer/__init__.py`
- Create: `our_work/pretrain/trainer/metrics.py`
- Create: `our_work/pretrain/scripts/run_pretrain.py`
- Create: `our_work/pretrain/scripts/run_eval.py`
- Test: `tests/our_work/pretrain/test_training_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
from pathlib import Path

from our_work.pretrain.scripts.run_pretrain import build_trainer_components


def test_build_trainer_components_returns_model_and_collator(tmp_path: Path):
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text('["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10"]', encoding="utf-8")
    components = build_trainer_components(
        model_config={"vocab_size": 5, "spectrum_dim": 2048, "prefix_length": 2, "n_positions": 16, "n_embd": 16, "n_layer": 1, "n_head": 2, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2},
        token_list=["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Ge_10"],
    )
    assert components["model"] is not None
    assert components["collator"] is not None
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `pytest tests/our_work/pretrain/test_training_smoke.py -v`  
Expected: FAIL with `ImportError` for missing pretrain script

- [ ] **Step 3: Add the YAML configs**

```yaml
# our_work/pretrain/configs/model/base_gpt.yaml
model:
  spectrum_dim: 2048
  prefix_length: 8
  n_positions: 32
  n_embd: 256
  n_layer: 6
  n_head: 8
```

```yaml
# our_work/pretrain/configs/train/base_train.yaml
training:
  output_dir: our_work/pretrain/outputs/base_run
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  num_train_epochs: 1
  learning_rate: 5.0e-4
  logging_steps: 10
  eval_strategy: steps
  eval_steps: 50
  save_steps: 50
```

- [ ] **Step 4: Implement the Trainer build entrypoint**

```python
# our_work/pretrain/scripts/run_pretrain.py
from __future__ import annotations

from transformers import Trainer, TrainingArguments

from our_work.pretrain.dataset.collator import SpectralCausalCollator
from our_work.pretrain.dataset.tokenizer import SpectralStructureTokenizer
from our_work.pretrain.model.configuration_spectral_gpt import SpectralGPTConfig
from our_work.pretrain.model.modeling_spectral_gpt import SpectralGPTForCausalLM


def build_trainer_components(model_config: dict, token_list: list[str]) -> dict:
    tokenizer = SpectralStructureTokenizer(tokens=token_list)
    config = SpectralGPTConfig(**model_config)
    model = SpectralGPTForCausalLM(config)
    collator = SpectralCausalCollator(tokenizer=tokenizer, prefix_length=config.prefix_length)
    return {"tokenizer": tokenizer, "model": model, "collator": collator}


def build_trainer(model, args: TrainingArguments, train_dataset, eval_dataset, collator) -> Trainer:
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
```

- [ ] **Step 5: Add the evaluation script and run smoke tests**

```python
# our_work/pretrain/scripts/run_eval.py
from __future__ import annotations

import torch

from our_work.pretrain.model.generation import generate_structure_tokens


@torch.inference_mode()
def run_eval_sample(model, tokenizer, spectra: torch.Tensor, max_new_tokens: int = 10) -> list[list[str]]:
    return generate_structure_tokens(model, tokenizer, spectra, max_new_tokens=max_new_tokens)
```

Run: `pytest tests/our_work/pretrain/test_training_smoke.py -v`  
Expected: PASS

Run: `pytest tests/our_work/pretrain -v`  
Expected: all pretrain tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/our_work/pretrain our_work/pretrain
git commit -m "feat: add spectral pretraining trainer stack"
```

---

## Self-Review

### Spec coverage
- 独立 tokenizer：Task 1
- shard 数据加载与 collator：Task 2
- decoder-only + projector：Task 3
- generation + teacher-forcing scoring：Task 4
- Trainer + HF 风格训练入口：Task 5

### Placeholder scan
- 未使用 `TODO/TBD`
- 每个实现步骤都有明确文件路径与代码
- 每个任务都有失败测试、通过测试与提交步骤

### Type consistency
- `spectra` 始终是 `(batch, 2048)` `float32`
- `input_ids` 只表示结构 token 区域，不包含 prefix
- `labels` 始终包含 prefix 区域并用 `-100` 屏蔽
- 模型类统一命名为 `SpectralGPTConfig` / `SpectralGPTForCausalLM`
