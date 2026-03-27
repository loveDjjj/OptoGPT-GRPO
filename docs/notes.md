# Notes

## 需求
继续优化 SFT 主链中的 `generation.py` 和 `scoring.py` 热路径，减少 Python 循环与显存重复分配，并让现有 batch 进度条显示当前子阶段。

## 修改文件
- models/optogpt/generation.py
- models/optogpt/scoring.py
- trainers/spectral_sft_trainer.py
- evaluators/spectrum_evaluator.py
- docs/notes.md
- docs/logs/2026-03.md

## 修改内容
- `generation.py`
  - 自回归解码改为预分配整块 `ys` 张量，避免每一步 `torch.cat`
  - 逐步 token 处理由“每一步 Python for 循环”改成张量式长度、结束标记与 logprob 累积
  - 新增后续 mask 缓存，避免反复构造相同长度的下三角 mask
- `scoring.py`
  - teacher forcing 的 `tgt_input/target_ids/token_mask` 改成批量扁平化回填，避免每条样本单独 `torch.tensor(...)`
  - 使用缓存后的下三角 mask
  - 无梯度打分改为 `torch.inference_mode()`
  - 结果张量改为一次性预分配后按 chunk 回填
- `spectral_sft_trainer.py` 与 `spectrum_evaluator.py`
  - 在现有 `tqdm` batch 进度条上增加 `stage` 字段
  - 训练阶段显示 `generate / tmm / score / backward`
  - 评测阶段显示 `score / generate / tmm / write`

## 验证
```bash
D:\anaconda\envs\oneday\python.exe -m compileall trainers evaluators losses runners models datasets utils
```

结果：通过

```bash
@'
import torch
from models.optogpt.scoring import sequence_logprobs_multi_target_batch_tensor

class DummyModel:
    def __init__(self):
        self.device = torch.device('cpu')
        self.pad_token = 'PAD'
        self.eos_token = 'EOS'
        self.bos_token = 'BOS'
        self.struc_word_dict = {'PAD': 0, 'EOS': 1, 'BOS': 2}
        self.spec_dim = 4
        self.max_len = 8
        self.model = self
        self.raw_model = self
    def prompt_ids(self, start_symbol='BOS', start_mat=None):
        return [2]
    def targets_to_tensor_batch(self, spectra):
        if torch.is_tensor(spectra):
            return spectra.to(dtype=torch.float32).unsqueeze(1)
        return torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
    def generator(self, out):
        return out
    def __call__(self, src, tgt, src_mask, tgt_mask):
        batch, tgt_len = tgt.shape
        vocab = 5
        logits = torch.zeros((batch, tgt_len, vocab), dtype=torch.float32)
        return logits

model = DummyModel()
logprobs, mask = sequence_logprobs_multi_target_batch_tensor(
    model=model,
    target_spectra=torch.zeros((3, 4), dtype=torch.float32),
    token_id_groups=[[3, 1], [4, 3, 1], [1]],
    require_grad=False,
    batch_size=2,
)
print(tuple(logprobs.shape), tuple(mask.shape), mask.sum().item())
'@ | D:\anaconda\envs\oneday\python.exe -
```

结果：通过

```bash
@'
import torch
from models.optogpt.generation import DecodeConfig, generate_structures_for_targets

class DummyModel:
    def __init__(self):
        self.device = torch.device('cpu')
        self.pad_token = 'PAD'
        self.eos_token = 'EOS'
        self.bos_token = 'BOS'
        self.struc_word_dict = {'PAD': 0, 'EOS': 1, 'BOS': 2, 'A': 3}
        self.struc_index_dict = {v: k for k, v in self.struc_word_dict.items()}
        self.max_len = 5
        self.model = self
    def prompt_ids(self, start_symbol='BOS', start_mat=None):
        return [2]
    def targets_to_tensor_batch(self, spectra):
        if torch.is_tensor(spectra):
            return spectra.to(dtype=torch.float32).unsqueeze(1)
        return torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
    def token_id_to_str(self, token_id):
        return self.struc_index_dict[int(token_id)]
    def token_ids_to_structure_tokens(self, token_ids, stop_at_eos=True):
        out = []
        for token_id in token_ids:
            token = self.token_id_to_str(int(token_id))
            if token == 'EOS' and stop_at_eos:
                break
            if token in {'PAD', 'BOS', 'UNK'}:
                continue
            out.append(token)
        return out
    def generator(self, out):
        logits = torch.full((out.shape[0], 4), -10.0)
        logits[:, 1] = 10.0
        return logits
    def __call__(self, src, tgt, src_mask, tgt_mask):
        return torch.zeros((src.shape[0], tgt.shape[1], 4), dtype=torch.float32)

model = DummyModel()
items = generate_structures_for_targets(
    model=model,
    target_spectra=torch.zeros((2, 4), dtype=torch.float32),
    decode_config=DecodeConfig(decode='greedy', batch_size=2, max_len=5),
    num_samples_per_target=1,
    target_indices=[10, 11],
)
print(len(items), items[0].token_ids, items[0].terminated_by_eos, items[0].max_len_reached)
'@ | D:\anaconda\envs\oneday\python.exe -
```

结果：通过

## Git
- branch: `perf/generation-scoring-hotpath`
- commit: `git commit -m "perf: optimize generation and scoring hot paths"`
