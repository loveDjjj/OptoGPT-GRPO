# OptoGPT Spectral GRPO

## 椤圭洰瀹氫綅
鏈」鐩綋鍓嶄繚鐣欎袱鏉′富绾匡細

- `鍏夎氨璇勬祴`
- `鍩轰簬鍏夎氨 reward 鐨?spectral GRPO 璁粌`

鍩哄骇妯″瀷鏄?[model/optogpt.pt](/O:/Optics%20Code/OptoGPT-GRPO/model/optogpt.pt)锛屽畠鏈韩宸茬粡鏄敤 `CE/SFT` 棰勮缁冨ソ鐨?OptoGPT銆? 
褰撳墠璁粌璺緞鍦ㄨ鍩哄骇涓婄户缁仛鍩轰簬鐩爣鍏夎氨鐨?group-relative policy optimization銆?

## 褰撳墠鐩綍
- `configs/eval/`
  鍏夎氨璇勬祴閰嶇疆銆?
- `configs/grpo/`
  鍏夎氨 GRPO 璁粌閰嶇疆銆?
- `runners/`
  杩愯鍏ュ彛锛沗run_grpo.py` 涓哄綋鍓嶈缁冧富鍏ュ彛銆?
- `models/optogpt/`
  鍩哄骇妯″瀷鍔犺浇銆佺敓鎴愩€乸olicy 瀹氫箟銆乼eacher forcing / policy-aware 鎵撳垎銆乧heckpoint 瀵煎嚭銆?
- `datasets/`
  鍏夎氨-缁撴瀯鎴愬鏁版嵁闆嗐€佸垏鍒嗕笌鍒嗗竷寮?sampler銆?
- `evaluators/`
  鍏夎氨璇勬祴閫昏緫涓庢寚鏍囪仛鍚堛€?
- `trainers/`
  GRPO 璁粌鍣ㄣ€?
- `losses/`
  搴忓垪鎹熷け銆丟RPO 鐩爣涓庡厜璋辨崯澶便€?
- `physics/`
  鍘?`TMM/` 妯″潡鏁翠綋杩佺Щ鍚庣殑鐗╃悊璁＄畻浠ｇ爜銆?
- `data/materials/`
  鏉愭枡搴撱€?
- `dataset/`
  褰撳墠浣跨敤鐨?`Spectrum_*.npy` 涓?`Structure_*.npy`銆?
- `core/`
  鏃?checkpoint 鍏煎灞傦紝淇濈暀浣嗕笉鎵╁睍鏂伴€昏緫銆?

## 鏁版嵁璇存槑
褰撳墠榛樿浣跨敤锛?

- 璁粌闆嗭細
  [dataset/Spectrum_train.npy](/O:/Optics%20Code/OptoGPT-GRPO/dataset/Spectrum_train.npy)
  [dataset/Structure_train.npy](/O:/Optics%20Code/OptoGPT-GRPO/dataset/Structure_train.npy)
- 楠岃瘉闆嗭細
  [dataset/Spectrum_test.npy](/O:/Optics%20Code/OptoGPT-GRPO/dataset/Spectrum_test.npy)
  [dataset/Structure_test.npy](/O:/Optics%20Code/OptoGPT-GRPO/dataset/Structure_test.npy)

濡傛灉鍚庣画闇€瑕佷弗鏍煎垝鍒?`train/val/test`锛屽彲浠ワ細

- 鐩存帴鏂板鐙珛 `val` 鏂囦欢
- 鎴栧湪閰嶇疆閲屽惎鐢?`data.val_ratio`

## 鍏ュ彛
### 1. 鍏夎氨璇勬祴
鍔熻兘锛?

- 杈撳叆鐩爣鍏夎氨
- 鐢熸垚缁撴瀯
- 璁＄畻鐪熷疄缁撴瀯鐨勫簭鍒楁崯澶?
- 璁＄畻鐢熸垚缁撴瀯瀵瑰簲鐨勫厜璋辨崯澶?
- 杈撳嚭鏍锋湰绾х粨鏋滀笌姹囨€荤粺璁?

鍛戒护锛?

```bash
python runners/run_spectrum_eval.py --config configs/eval/spectrum_eval.yaml
```

澶氬崱锛?

```bash
torchrun --nproc_per_node=4 runners/run_spectrum_eval.py --config configs/eval/spectrum_eval.yaml
```

### 2. 鍏夎氨 GRPO 璁粌
鍔熻兘锛?

- 瀵规瘡鏉＄洰鏍囧厜璋?rollout 閲囨牱涓€缁勭粨鏋勫€欓€?
- 鐢ㄥ悓涓€ policy 瀹氫箟璁板綍 old logprobs
- 鐢?TMM 璁＄畻姣忎釜鍊欓€夌粨鏋勭殑鍏夎氨 loss锛屽苟杞垚 reward
- 鍦ㄥ悓涓€ target spectrum 鐨勭粍鍐呭仛 reward 涓績鍖?/ 鏍囧噯鍖?advantage
- 鐢?PPO-style clipped objective 鏇存柊妯″瀷

鍛戒护锛?

```bash
python runners/run_grpo.py --config configs/grpo/spectral_grpo.yaml
```

澶氬崱锛?

```bash
torchrun --nproc_per_node=4 runners/run_grpo.py --config configs/grpo/spectral_grpo.yaml
```

## 澶氬崱寤鸿
褰撳墠妯″瀷瑙勬ā涓嶅ぇ锛屾渶鍚堥€傜殑骞惰鏂瑰紡鏄?`DDP 鏁版嵁骞惰`锛屼笉鏄ā鍨嬪苟琛屻€?

- 寮€鍙戣皟璇曪細`1-2 鍗
- 姝ｅ紡璁粌锛歚4 鍗閫氬父鏈€鍧囪　
- 澶ц妯¤瘎娴嬶細`4-8 鍗閮藉彲浠?
- 璁粌闃舵榛樿璺宠繃 `Structure_train.npy` 鍔犺浇锛岄伩鍏嶆瘡涓?rank 閲嶅鍗犵敤澶у潡涓绘満鍐呭瓨
- rollout / scoring / TMM 閮藉敖閲忔寜澶?batch 鎵瑰鐞嗭紝浼樺厛鎻愰珮 GPU 鍒╃敤鐜囦笌鍚炲悙
- 濡傛灉瑕侀暱鏈熻窇 `4-8 鍗锛屾洿鎺ㄨ崘 `Linux + NCCL`锛涘綋鍓?Windows 鐜浼氶€€鍥炲埌 `Gloo`

## 杈撳嚭鐩綍
鍏夎氨璇勬祴杈撳嚭锛?

- `outputs/eval/<experiment>_<timestamp>/config.snapshot.yaml`
- `outputs/eval/<experiment>_<timestamp>/metrics/*.csv`
- `outputs/eval/<experiment>_<timestamp>/samples/*.jsonl`
- `outputs/eval/<experiment>_<timestamp>/plots/<split>/rankXX/*.png`
- `outputs/eval/<experiment>_<timestamp>/plots/summary/<split>_distribution.png`

鍏夎氨 GRPO 璁粌杈撳嚭锛?

- `outputs/grpo/<experiment>_<timestamp>/config.snapshot.yaml`
- `outputs/grpo/<experiment>_<timestamp>/metrics/*.csv`
- `outputs/grpo/<experiment>_<timestamp>/checkpoints/best.pt`
- `outputs/grpo/<experiment>_<timestamp>/checkpoints/final.pt`

## 渚濊禆
杩愯鍓嶈纭浠ヤ笅渚濊禆鍙敤锛?

- `python`
- `torch`
- `PyYAML`
- `numpy`
- `scipy`

鍙€夛細

- `matplotlib`
- `tqdm`
  `pso` 鐨勮ˉ鍏呮暟鎹泦鎼滅储浼氱敤瀹冩樉绀?target/layer 绾у埆鐨勮繘搴︽潯锛涙湭瀹夎鏃朵細閫€鍖栦负鏅€氳緭鍑恒€?
- `tensorboard`
  `pretrain` 鐨勫疄鏃舵崯澶便€佸涔犵巼銆佹搴﹁寖鏁扮瓑鍙鍖栭粯璁ら€氳繃 TensorBoard 鏌ョ湅銆?
  `data_gen` 鑷姩鍒嗘瀽鍜岃瘎娴嬬粯鍥鹃兘浼氱敤鍒般€?

## our_work 鏈嶅姟鍣ㄩ儴缃蹭笌杩愯
鏈妭瀵瑰簲浠撳簱鏍圭洰褰曚笅鐨?[our_work](/O:/Optics%20Code/OptoGPT-GRPO/our_work) 鐙珛鏁版嵁鐢熸垚銆侀璁粌涓庤瘎娴嬮摼璺€傚綋鍓嶉粯璁ら厤缃凡缁忔敼鎴愭湇鍔″櫒鍙洿鎺ヨ繍琛岀殑鐗堟湰锛屼笉鍐嶄緷璧栧繀椤讳粠浠撳簱鏍圭洰褰曞惎鍔紱涓嶈繃涓轰簡鎺掓煡鏃ュ織鍜屼骇鐗╂洿鐩磋锛屼粛鐒跺缓璁厛 `cd` 鍒颁粨搴撴牴鐩綍鍐嶆墽琛屻€?

### 1. 蹇呴』鍚屾鐨勭洰褰?
- 浠撳簱浠ｇ爜鏈韩
  - `git clone` 鎴?`git pull` 鍗冲彲锛宍our_work/` 宸茬粡鍦ㄤ富宸ヤ綔鍖烘牴鐩綍銆?
- `database/`
  - 杩欐槸鏉愭枡搴擄紝褰撳墠涓嶅湪 git 涓€?
  - 鏈嶅姟鍣ㄥ彧 `git clone` 涓嶅锛屽繀椤诲崟鐙悓姝ャ€?

### 2. 鏈嶅姟鍣ㄤ緷璧?
杩愯 `our_work` 閾捐矾鍓嶈纭浠ヤ笅 Python 渚濊禆鍙敤锛?

- `torch`
- `PyYAML`
- `numpy`
- `scipy`
- `pandas`
- `pyarrow`
- `openpyxl`
- `transformers`
- `safetensors`
- `Pillow`
- `tqdm`

鎺ㄨ崘瀹夎绀轰緥锛?

```bash
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas pyarrow openpyxl pyyaml pillow transformers safetensors tqdm
```

瀹夎瀹屾垚鍚庡彲鎵ц锛?

```bash
python -c "import torch,yaml,pandas,scipy,numpy,PIL,transformers,tqdm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

鍏稿瀷缁堢杈撳嚭锛?

```text
torch 2.x.x cuda True
```

### 3. 榛樿閰嶇疆涓庡叧閿害鏉?
榛樿鏈嶅姟鍣ㄩ厤缃枃浠讹細

- 鏁版嵁鐢熸垚锛歔dataset_v1.yaml](/O:/Optics%20Code/OptoGPT-GRPO/data_gen/configs/dataset_v1.yaml)
- 鏁版嵁鐢熸垚锛? 鍗★級锛歚data_gen/configs/a100_4gpu.yaml`
- 鏁版嵁鐢熸垚锛? 鍗★級锛歚data_gen/configs/a100_8gpu.yaml`
- 璁粌锛歔base_train.yaml](/O:/Optics%20Code/OptoGPT-GRPO/pretrain/configs/train/base_train.yaml)
- 璁粌锛? 鍗★級锛歚pretrain/configs/train/a100_4gpu.yaml`
- 璁粌锛? 鍗★級锛歚pretrain/configs/train/a100_8gpu.yaml`
- 妯″瀷锛歔base_gpt.yaml](/O:/Optics%20Code/OptoGPT-GRPO/pretrain/configs/model/base_gpt.yaml)
- 寮哄寲瀛︿範锛堝熀纭€锛夛細`rl/configs/grpo/base_grpo.yaml`
- 寮哄寲瀛︿範锛? 鍗★級锛歚rl/configs/grpo/a100_4gpu.yaml`
- 寮哄寲瀛︿範锛? 鍗★級锛歚rl/configs/grpo/a100_8gpu.yaml`
- PSO 琛ュ厖鏁版嵁闆嗭細`pso/configs/pso_supplement.yaml`
- GA 浼樼瑙ｆ棌琛ュ厖鏁版嵁闆嗭細`ga/configs/ga_seeded_absorbers.yaml`

褰撳墠榛樿鍊硷紙鍗曞崱 A100 80G + 16 CPU锛夛細

- `dataset_v1.yaml`
  - `paths.database_dir: database`
  - `paths.output_dir: outputs/our_work/data_gen/v1`
  - `data.layer_counts: [5, 6, 7, 8, 9, 10]`
  - `data.samples_per_bucket: 500000`
  - `data.thickness_range_nm: {min: 10, max: 500, step: 10}`
  - `sampling.device: auto`
  - `sampling.batch_size: 65536`
  - `sampling.max_duplicate_retry: 1000`
  - `tmm.device: auto`
  - `tmm.cpu_threads: 16`
  - `tmm.batch_size: 4096`
  - `tmm.num_points: 1024`
  - `analysis.enabled: true`
  - `analysis.auto_after_build: true`
  - `analysis.scopes: [all]`
  - `analysis.spectrum.pca_components: 8`
  - `analysis.spectrum.cluster_count: 16`
- `base_train.yaml`
  - `data.dataset_dir: outputs/our_work/data_gen/v1`
  - `data.vocab_path: outputs/our_work/data_gen/v1/vocab/vocab.json`
  - `data.num_workers: 8`
  - `data.prefetch_factor: 4`
  - `data.pin_memory: true`
  - `data.persistent_workers: true`
  - `training.output_dir: outputs/our_work/pretrain/base_train`
  - `training.per_device_train_batch_size: 16`
  - `training.per_device_eval_batch_size: 64`
  - `training.gradient_accumulation_steps: 2`
  - `training.max_steps: null`
  - `training.num_train_epochs: 5`
  - `training.learning_rate: 1e-4`
  - `training.bf16: true`
  - `training.tf32: true`
  - `training.logging_steps: 1000`
  - `training.eval_steps: 100000`
  - `training.save_steps: 50000`
  - `monitoring.tensorboard/jsonl/csv/save_plots: true`
  - 璇勪及璺緞浼氬厛鎶?logits 棰勫鐞嗘垚 `argmax token ids` 鍐嶅仛 metrics锛岄伩鍏嶅畬鏁存敹闆?`[batch, seq_len, vocab]` 绾у埆鐨勫ぇ寮犻噺
  - `distributed.*` 鍙湁鍦?`torchrun --nproc_per_node=...` 鐨勭湡瀹炲鍗＄幆澧冧笅鎵嶄細鐢熸晥锛涘崟杩涚▼ `python run_pretrain.py ...` 浼氬拷鐣ヨ繖閮ㄥ垎骞舵竻鐞嗚剰鐨?DDP 鐜鍙橀噺
- `base_gpt.yaml`
  - `model.spectrum_dim: 2048`
  - `model.prefix_length: 8`
  - `model.n_embd: 1024`
  - `model.n_layer: 6`
  - `model.n_head: 16`
- `base_grpo.yaml`
  - `training.per_device_batch_size: 16`
  - `rollout.group_size: 4`
  - `rollout.batch_size: 512`
  - `scoring.batch_size: 1024`
  - `reward.tmm.batch_size: 4096`
  - `monitoring.tensorboard/jsonl/csv/save_plots: true`
- `pso_supplement.yaml`
  - `paths.database_dir: _shared/database`
  - `paths.output_dir: outputs/our_work/data_gen/pso_supplement`
  - `data.layer_counts: [5, 6, 7, 8, 9, 10]`
  - `data.thickness_range_nm: {min: 10, max: 500, step: 10}`
  - `targets.include_fixed: true`
  - `targets.include_lorentzian: true`
  - `targets.lorentzian.center_min_um: 2.1`
  - `targets.lorentzian.center_max_um: 14.9`
  - `targets.lorentzian.center_step_um: 0.1`
  - `targets.lorentzian.fwhm_um: 0.02`
  - `search.population_size: 8192`
  - `search.iterations: 50`
  - `search.batch_size: 2048`
  - `search.max_accepted_per_target_layer: 1000`
  - `search.acceptance_mse_threshold: 0.01`
  - `tmm.wavelength_range_um: [2.0, 15.0]`
  - `tmm.num_points: 1024`
  - `tmm.batch_size: 2048`
- `ga_seeded_absorbers.yaml`
  - `paths.database_dir: _shared/database`
  - `paths.output_dir: outputs/our_work/data_gen/ga_seeded_absorbers`
  - `data.max_samples_per_target: 100`
  - `data.thickness_range_nm: {min: 10, max: 500, step: 10}`
  - `data.include_seed_thickness_values: true`
  - `targets.tasks: 榛樿鏄惧紡鍐欏叆 3 涓?seeded 浠诲姟`
  - `targets.include_ids: null`
  - `targets.tasks[*].bands: 鍙湪澹版槑鐨勬尝娈佃绠?loss`
  - `targets.tasks[*].seed_tokens/random_init: 鏀寔缁欏弬鑰冪粨鏋勬垨闅忔満鍒濆鍖栫粨鏋刞
  - `ga_custom_tasks.yaml: 鐢ㄦ埛鑷畾涔変换鍔℃ā鏉縛
  - `search.population_size: 8192`
  - `search.generations_per_restart: 20`
  - `search.restart_count: 5`
  - `search.batch_size: 8192`
  - `search.acceptance_floor_mse: 0.005`
  - `tmm.wavelength_range_um: [2.0, 15.0]`
  - `tmm.num_points: 1024`
  - `visualization.enabled: true`

鍏抽敭绾︽潫锛?

- `model.spectrum_dim` 蹇呴』绛変簬 `2 * tmm.num_points`
- 濡傛灉浣犳妸 `num_points` 鏀规垚涓嶆槸 `1024`锛屽氨蹇呴』鍚屾淇敼 `model.spectrum_dim`
- 鐜板湪 YAML 鍐呯殑鐩稿璺緞閮戒細鑷姩鎸変粨搴撴牴鐩綍瑙ｆ瀽

### 4. 浠庨浂寮€濮嬮儴缃蹭笌杩愯
浠ヤ笅姝ラ鍋囪鏈嶅姟鍣ㄩ儴缃茬洰褰曚负 `/srv/OptoGPT-GRPO`锛屽苟涓旂湡瀹炴暟鎹緭鍑轰繚瀛樺湪浠撳簱鏍圭洰褰曠殑 `outputs/` 涓嬨€?

#### Step 1: 鎷変唬鐮佸苟杩涘叆浠撳簱

```bash
cd /srv
git clone <your-repo-url> OptoGPT-GRPO
cd /srv/OptoGPT-GRPO
git checkout main
```

鍏稿瀷缁堢杈撳嚭锛?

```text
Cloning into 'OptoGPT-GRPO'...
Already on 'main'
```

姝ゆ椂浣犲簲鑳界湅鍒帮細

- [our_work](/O:/Optics%20Code/OptoGPT-GRPO/our_work)
- [README.md](/O:/Optics%20Code/OptoGPT-GRPO/README.md)

#### Step 2: 鍚屾鏉愭枡搴?

鎶婃湰鍦?`database/` 鍚屾鍒版湇鍔″櫒浠撳簱鏍圭洰褰曪紝渚嬪锛?

```bash
scp -r database user@server:/srv/OptoGPT-GRPO/
```

鍚屾瀹屾垚鍚庢湇鍔″櫒涓婂簲瀛樺湪锛?

- `/srv/OptoGPT-GRPO/database/*.csv`
- 鎴?`/srv/OptoGPT-GRPO/database/*.xlsx`

浣犲彲浠ユ墽琛岋細

```bash
ls /srv/OptoGPT-GRPO/database | head
```

鍏稿瀷缁堢杈撳嚭锛?

```text
Ag.xlsx
Al.xlsx
Ge.xlsx
SiO2.xlsx
...
```

#### Step 3: 瀹夎渚濊禆

```bash
cd /srv/OptoGPT-GRPO
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas pyarrow openpyxl pyyaml pillow transformers safetensors
```

鍏稿瀷缁堢杈撳嚭锛?

```text
Successfully installed ...
```

#### Step 4: 鐢熸垚鏁版嵁闆?

杩愯鍓嶏紝寤鸿鍏堢‘璁ゆ暟鎹敓鎴愰厤缃噷纭疄鍚敤浜嗗垎鍧楅噰鏍峰拰鍒嗗潡 TMM锛?

```bash
cd /srv/OptoGPT-GRPO
python -c "import yaml, pathlib; cfg=yaml.safe_load(pathlib.Path('data_gen/configs/dataset_v1.yaml').read_text(encoding='utf-8')); print('sampling =', cfg['sampling']); print('tmm.batch_size =', cfg['tmm']['batch_size'])"
```

鍏稿瀷缁堢杈撳嚭锛?

```text
sampling = {'device': 'auto', 'batch_size': 65536, 'max_duplicate_retry': 1000}
tmm.batch_size = 4096
```

榛樿閰嶇疆鐗囨濡備笅锛?

```yaml
data:
  layer_counts: [5, 6, 7, 8, 9, 10]
  samples_per_bucket: 500000
  thickness_range_nm:
    min: 10
    max: 500
    step: 10

sampling:
  device: auto
  batch_size: 65536
  max_duplicate_retry: 1000

tmm:
  wavelength_range_um: [2.0, 15.0]
  num_points: 1024
  incident_angle: 0.0
  polarization: 0
  tolerance: 0.001
  complex_dtype: complex128
  batch_size: 4096

analysis:
  enabled: true
  auto_after_build: true
  batch_size: 8192
  scopes: [all]
  structure:
    enabled: true
  spectrum:
    enabled: true
    device: auto
    engine: rapids
    pca_components: 8
    pca_fit_samples: 50000
    cluster_count: 16
    cluster_fit_samples: 50000
    cluster_iterations: 20
    scatter_max_points: 20000
    save_split_analysis: false
```

璇存槑锛?
- 榛樿鑷姩鍒嗘瀽鍙窇 `all`锛岄伩鍏嶇敓鎴愬畬鎴愬悗鍐嶆妸 `train / val / test` 閲嶅鎵竴閬嶃€?
- 褰?`analysis.spectrum.engine: rapids` 鏃讹紝`run_build_dataset.py` 浼氬湪鐙珛瀛愯繘绋嬮噷璋冪敤 `run_analyze_dataset.py`锛岄伩鍏嶅悓涓€ Python 杩涚▼閲屾贩鐢?`torch` 鍜?RAPIDS 鐨?CUDA 杩愯鏃舵爤銆?

```bash
cd /srv/OptoGPT-GRPO
python data_gen/scripts/run_build_dataset.py --config data_gen/configs/dataset_v1.yaml
```

鍏稿瀷缁堢杈撳嚭锛?

```text
data_gen buckets:  17%|鈻堚枊        | 1/6 [00:xx<00:xx, ... bucket/s, layer_count=5, bucket_kept=98304, bucket_target=500000, sample_batch=65536, tmm_batch=4096, duplicates_skipped=..., valid_kept=...]
```

璇ユ楠ゅ畬鎴愬悗搴斿嚭鐜帮細

- `outputs/our_work/data_gen/v1/shards/shard-00000.parquet`
- `outputs/our_work/data_gen/v1/splits/split_manifest.json`
- `outputs/our_work/data_gen/v1/vocab/vocab.json`
- `outputs/our_work/data_gen/v1/analysis/all/structure_material_by_layer.png`
- `outputs/our_work/data_gen/v1/analysis/all/structure_thickness_by_layer.png`
- `outputs/our_work/data_gen/v1/analysis/all/spectrum_mean_std.png`
- `outputs/our_work/data_gen/v1/analysis/all/spectrum_pca_scatter.png`
- `outputs/our_work/data_gen/v1/analysis/all/spectrum_cluster_sizes.png`
- `outputs/our_work/data_gen/v1/analysis/all/spectrum_cluster_representatives.png`

璇存槑锛?

- 缁撴瀯鍊欓€夌幇鍦ㄦ寜 `sampling.batch_size` 鍦?GPU/CPU 涓婂垎鍧楃敓鎴愩€?
- TMM 鍏夎氨璁＄畻鎸?`tmm.batch_size` 鍒嗘壒鎵ц锛屼笉浼氬啀鎶婃暣 bucket 涓€娆℃€ч€佽繘鏄惧瓨/鍐呭瓨銆?
- bucket 鍐呬粛鐒朵繚鎸佸叏灞€涓ユ牸鍞竴锛涢噸澶嶇粨鏋勪細琚涪寮冨苟鑷姩琛ラ噰銆?
- 鏁版嵁鐢熸垚缁撴潫鍚庨粯璁ゅ彧鑷姩璺?`all` 鍒嗘瀽锛沗train/val/test` 寤鸿閫氳繃鐙珛 CLI 鎸夐渶琛ヨ窇銆?
- 褰撳墠榛樿鍙嚜鍔ㄥ垎鏋?`all`锛岄伩鍏嶅 `train/val/test` 閲嶅鎵弿瀵艰嚧鑰楁椂杩囬暱銆?
- 鍏夎氨鍒嗘瀽浣跨敤鎷兼帴鍚庣殑 `[R..., T...]` 鍋氭爣鍑嗗寲銆丳CA 鍜岃仛绫伙紱缁撴瀯鍒嗘瀽浼氭妸鏉愭枡鍜屽帤搴︽媶寮€缁熻銆?
- 鍏夎氨鍒嗘瀽浼樺厛璧?RAPIDS锛坄cudf + cuml`锛夛紝鎶?PCA / 鑱氱被涓昏矾寰勬斁鍦?GPU 涓娿€?

浣犲彲浠ユ鏌ワ細

```bash
ls outputs/our_work/data_gen/v1/shards | head
cat outputs/our_work/data_gen/v1/splits/split_manifest.json
ls outputs/our_work/data_gen/v1/analysis/all
```

濡傛灉浣犲垏鍒板鍗★紝鐩存帴浣跨敤涓撶敤閰嶇疆锛?

4 鍗?A100 姝ｅ紡鍛戒护锛?

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=4 data_gen/scripts/run_build_dataset.py --config data_gen/configs/a100_4gpu.yaml
```

8 鍗?A100 姝ｅ紡鍛戒护锛?

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=8 data_gen/scripts/run_build_dataset.py --config data_gen/configs/a100_8gpu.yaml
```

璇存槑锛?

- 褰撳墠澶氬崱鏁版嵁鐢熸垚鍏堟寜 `layer bucket` 鍦?rank 涔嬮棿鍒嗛厤锛屼繚璇?bucket 鍐呭叏灞€鍞竴涓嶄細琚法 rank 鐮村潖銆?
- `4` 鍗℃椂 6 涓?bucket 浼氬垎鍒?4 涓?rank銆?
- `8` 鍗℃椂浼氭湁绌洪棽 rank锛岃繖鏄綋鍓嶇増鏈负浜嗕繚璇佸敮涓€鎬у拰姝ｇ‘鎬у仛鐨勪繚瀹堝疄鐜般€?

#### Step 4.1: 鍗曠嫭杩愯鏁版嵁闆嗗垎鏋?

濡傛灉浣犲凡缁忔湁鐜版垚鏁版嵁闆嗭紝涔熷彲浠ヤ笉閲嶆柊鐢熸垚锛岀洿鎺ュ崟鐙窇鍒嗘瀽锛?

```bash
cd /srv/OptoGPT-GRPO
python data_gen/scripts/run_analyze_dataset.py \
  --dataset-dir outputs/our_work/data_gen/v1 \
  --split all \
  --wavelength-min 2.0 \
  --wavelength-max 15.0 \
  --device auto
```

鍏稿瀷缁堢杈撳嚭锛?

```text
# 鍛戒护鏈韩榛樿瀹夐潤鎵ц锛屽畬鎴愬悗浼氬湪 analysis 鐩綍涓嬪啓鍑?PNG / JSON 缁撴灉
```

濡傛灉鍙垎鏋愭煇浜?shard 鏂囦欢锛?

```bash
cd /srv/OptoGPT-GRPO
python data_gen/scripts/run_analyze_dataset.py \
  --shard-path outputs/our_work/data_gen/v1/shards/shard-00000.parquet \
  --shard-path outputs/our_work/data_gen/v1/shards/shard-00001.parquet \
  --output-dir outputs/our_work/data_gen/custom_analysis \
  --wavelength-min 2.0 \
  --wavelength-max 15.0 \
  --device cpu
```

濡傛灉浣犺繕鎯冲崟鐙垎鏋?`train / val / test`锛岀洿鎺ユ敼 `--split` 鍗冲彲锛屼緥濡傦細

```bash
cd /srv/OptoGPT-GRPO
python data_gen/scripts/run_analyze_dataset.py \
  --dataset-dir outputs/our_work/data_gen/v1 \
  --split train \
  --wavelength-min 2.0 \
  --wavelength-max 15.0 \
  --device auto
```

璇ユ楠ゅ畬鎴愬悗搴斿嚭鐜帮細

- `outputs/our_work/data_gen/v1/analysis/analysis_manifest.json`
- `outputs/our_work/data_gen/v1/analysis/<scope>/structure_analysis.json`
- `outputs/our_work/data_gen/v1/analysis/<scope>/spectrum_analysis.json`
- 瀵瑰簲 scope 涓嬬殑缁撴瀯鍒嗗竷鍜岃氨褰㈠垎鏋?PNG

#### Step 4.2: 杞崲骞跺垎鏋愭棫 `.npy` 鏁版嵁闆?

鏃ф暟鎹泦鏂囦欢锛?

- `dataset/Spectrum_train.npy`
- `dataset/Spectrum_test.npy`
- `dataset/Structure_train.npy`
- `dataset/Structure_test.npy`

涓嶈兘鐩存帴浼犵粰 `data_gen/scripts/run_analyze_dataset.py`锛岄渶瑕佸厛杞崲鎴?`data_gen` 鐨?parquet schema銆?

杞崲鍛戒护锛?

```bash
cd /srv/OptoGPT-GRPO
python -m data_gen.scripts.convert_legacy_npy_dataset \
  --spectrum-train dataset/Spectrum_train.npy \
  --structure-train dataset/Structure_train.npy \
  --spectrum-test dataset/Spectrum_test.npy \
  --structure-test dataset/Structure_test.npy \
  --output-dir outputs/legacy_npy_parquet \
  --records-per-shard 50000 \
  --num-workers 8
```

璇ユ楠ゅ畬鎴愬悗搴斿嚭鐜帮細

- `outputs/legacy_npy_parquet/shards/train-shard-00000.parquet`
- `outputs/legacy_npy_parquet/shards/test-shard-00000.parquet`
- `outputs/legacy_npy_parquet/splits/split_manifest.json`
- `outputs/legacy_npy_parquet/vocab/vocab.json`
- `outputs/legacy_npy_parquet/stats/summary.json`

璇存槑锛?

- `Spectrum_*.npy` 浼氭寜琛屽鍒跺埌 `spectrum_rt` 瀛楁銆?
- `Structure_*.npy` 浼氫粠 `Material_ThicknessNm` token 鎷嗗嚭 `materials` 鍜?`thickness_nm`銆?
- 鏃ф暟鎹泦鐨勫厜璋辩淮搴﹂€氬父鏄?`142 = R(71) + T(71)`锛屽搴旀棫閰嶇疆 `0.4-1.1 um`銆乣71` 涓尝闀跨偣銆?
- `Structure_train.npy` 鏄?object array锛孨umPy 涓嶈兘鍐呭瓨鏄犲皠锛涜浆鎹㈣剼鏈細涓€娆″彧鍔犺浇涓€涓?split锛屼絾杞崲 train 鏃朵粛闇€瑕佹湇鍔″櫒鏈夎冻澶熷唴瀛樺绾宠 object 鏁扮粍銆?
- `--num-workers` 榛樿涓?`1`銆傚綋璁剧疆涓哄ぇ浜?`1` 鏃讹紝鑴氭湰浼氭寜 shard 澶氳繘绋嬪苟琛屽啓 parquet锛涘苟琛屽墠浼氬厛鎵弿缁撴瀯 token 鏋勫缓绋冲畾 vocab銆?
- 澶氳繘绋嬫ā寮忎笅锛屾瘡涓?worker 杩涚▼閮戒細鍔犺浇褰撳墠 split 鐨?`Structure_*.npy` object array锛屽洜姝や富鏈哄唴瀛樺崰鐢ㄤ細杩戜技闅?`num_workers` 鏀惧ぇ锛涘鏋滃唴瀛樼揣寮狅紝鍏堜粠 `4` 鎴栨洿灏忓€煎紑濮嬨€?

濡傛灉鍙兂鍏堝皬瑙勬ā楠岃瘉杞崲娴佺▼锛屽彲浠ュ姞閲囨牱涓婇檺锛?

```bash
cd /srv/OptoGPT-GRPO
python -m data_gen.scripts.convert_legacy_npy_dataset \
  --spectrum-train dataset/Spectrum_train.npy \
  --structure-train dataset/Structure_train.npy \
  --spectrum-test dataset/Spectrum_test.npy \
  --structure-test dataset/Structure_test.npy \
  --output-dir outputs/legacy_npy_parquet_smoke \
  --records-per-shard 50000 \
  --max-train-samples 10000 \
  --max-test-samples 10000 \
  --num-workers 2
```

杞崲鍚庤繍琛屽垎鏋愶細

```bash
cd /srv/OptoGPT-GRPO
python data_gen/scripts/run_analyze_dataset.py \
  --dataset-dir outputs/legacy_npy_parquet \
  --scope train \
  --scope test \
  --output-dir outputs/legacy_npy_analysis \
  --wavelength-min 0.4 \
  --wavelength-max 1.1 \
  --engine rapids \
  --device auto
```

濡傛灉鏈嶅姟鍣ㄦ病鏈?RAPIDS / cudf / cuml锛屽彧鍒嗘瀽缁撴瀯鍒嗗竷锛?

```bash
cd /srv/OptoGPT-GRPO
python data_gen/scripts/run_analyze_dataset.py \
  --dataset-dir outputs/legacy_npy_parquet \
  --scope train \
  --scope test \
  --output-dir outputs/legacy_npy_analysis_structure_only \
  --wavelength-min 0.4 \
  --wavelength-max 1.1 \
  --disable-spectrum-analysis
```

#### Step 4.3: 鐢熸垚 PSO 琛ュ厖鏁版嵁闆?

PSO 琛ュ厖鏁版嵁闆嗙敤浜庡洿缁曟寚瀹氱洰鏍囧惛鏀惰氨鎼滅储鐩歌繎缁撴瀯锛屼綔涓洪殢鏈虹敓鎴愭暟鎹泦涔嬪鐨勫畾鍚戣ˉ鍏呮暟鎹€傞粯璁ょ洰鏍囧寘鎷細

- `broad_3_13`锛歚3-13 um` 鍚告敹涓?1锛屽叾浣欎负 0銆?
- `band_5_8`锛歚5-8 um` 鍚告敹涓?1锛屽叾浣欎负 0銆?
- `dual_3_5_8_13`锛歚3-5 um` 鍜?`8-13 um` 鍚告敹涓?1锛屽叾浣欎负 0銆?
- `notch_3_5`锛歚3-5 um` 鍚告敹涓?0锛屽叾浣欎负 1銆?
- 娲涗鸡鍏圭獎甯︾洰鏍囷細`2.1-14.9 um`锛屼腑蹇冩闀?`0.1 um`锛屽崐楂樺 `0.02 um`銆?

鍗曡繘绋嬭繍琛岋細

```bash
cd /srv/OptoGPT-GRPO
python -m pso.scripts.run_pso_dataset --config pso/configs/pso_supplement.yaml
```

璇ユ楠ゅ畬鎴愬悗搴斿嚭鐜帮細

- `outputs/our_work/data_gen/pso_supplement/shards/shard-00000.parquet`
- `outputs/our_work/data_gen/pso_supplement/splits/split_manifest.json`
- `outputs/our_work/data_gen/pso_supplement/vocab/vocab.json`
- `outputs/our_work/data_gen/pso_supplement/targets/target_manifest.json`
- `outputs/our_work/data_gen/pso_supplement/stats/summary.json`
- `outputs/our_work/data_gen/pso_supplement/stats/search_summary.json`

璇存槑锛?

- 瀹夎 `tqdm` 鍚庯紝杩愯杩囩▼涓細鏄剧ず `pso rank <rank>/<world_size>` 鐨?target/layer 绾у埆杩涘害鏉°€?
- PSO 缁撴瀯鍙傛暟涓庝富鏁版嵁鐢熸垚閾捐矾淇濇寔涓€鑷达細`5-10` 灞傘€乣10-500 nm`銆佸帤搴︽闀?`10 nm`銆佹潗鏂欐潵鑷?`_shared/database/`銆?
- 杈撳嚭鍏夎氨浠嶇劧鏄?`[R..., T...]`锛屽叡 `2048` 缁达紱鐩爣鍚告敹璋卞彧鐢ㄤ簬 PSO 鎼滅储鏃惰绠?MSE銆?
- 鍙湁 `absorption MSE < search.acceptance_mse_threshold` 鐨勭粨鏋勪細琚啓鍏ユ暟鎹泦銆?
- 鍐欏嚭鍓嶄細鎸夊畬鏁?`structure_tokens` 鍋氬叏灞€鍘婚噸銆?
- 璇ヨˉ鍏呮暟鎹粯璁ゅ啓鍏ョ嫭绔嬬洰褰曪紝涓嶄細鑷姩娣峰叆闅忔満鏁版嵁闆嗭紱鍚庣画璁粌娣峰悎姣斾緥闇€瑕佸湪璁粌鏁版嵁鍔犺浇渚у崟鐙畾涔夈€?

澶氳繘绋嬫媶鍒嗚繍琛岋細

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=4 -m pso.scripts.run_pso_dataset --config pso/configs/pso_supplement.yaml
```

浣跨敤澶氳繘绋嬪墠锛岄渶瑕佸厛鎶?`pso/configs/pso_supplement.yaml` 閲岀殑 `distributed.enabled` 鏀规垚 `true`銆傚杩涚▼浼氭寜 `target/layer` work items 鎷嗗垎浠诲姟锛屽苟鍒嗗埆鍐欏埌锛?

- `outputs/our_work/data_gen/pso_supplement/rank00`
- `outputs/our_work/data_gen/pso_supplement/rank01`
- `outputs/our_work/data_gen/pso_supplement/rankXX`

褰撳墠鐗堟湰杩樻病鏈夊唴缃法 rank 鍚堝苟涓庝簩娆″幓閲嶈剼鏈紱姝ｅ紡娣峰叆璁粌鍓嶏紝寤鸿鍏堝鍚?`rankXX` 鐩綍鍋氬悎骞跺拰鍏ㄥ眬鍘婚噸銆?

#### Step 4.4: 鍒嗘瀽 PSO 琛ュ厖鏁版嵁闆?

PSO 鏁版嵁闆嗙敓鎴愬畬鎴愬悗锛屽彲浠ュ崟鐙繍琛屽垎鏋愬拰鍙鍖栵細

```bash
cd /srv/OptoGPT-GRPO
python -m pso.analysis.run_analyze_pso \
  --dataset-dir outputs/our_work/data_gen/pso_supplement \
  --output-dir outputs/our_work/pso_analysis/pso_supplement \
  --split all \
  --wavelength-min-um 2.0 \
  --wavelength-max-um 15.0 \
  --top-k 8 \
  --max-spectrum-groups 100
```

濡傛灉瑕佺粰鎵€鏈?`target/layer` 缁勫悎閮界敾鍏夎氨鍥撅紝鎶?`--max-spectrum-groups` 鏀规垚 `-1`锛?

```bash
cd /srv/OptoGPT-GRPO
python -m pso.analysis.run_analyze_pso \
  --dataset-dir outputs/our_work/data_gen/pso_supplement \
  --output-dir outputs/our_work/pso_analysis/pso_supplement_full \
  --split all \
  --max-spectrum-groups -1
```

璇ユ楠ゅ畬鎴愬悗搴斿嚭鐜帮細

- `outputs/our_work/pso_analysis/pso_supplement/summary.json`
- `outputs/our_work/pso_analysis/pso_supplement/analysis_manifest.json`
- `outputs/our_work/pso_analysis/pso_supplement/tables/target_layer_stats.csv`
- `outputs/our_work/pso_analysis/pso_supplement/tables/search_efficiency.csv`
- `outputs/our_work/pso_analysis/pso_supplement/tables/material_stats.csv`
- `outputs/our_work/pso_analysis/pso_supplement/tables/diversity_stats.csv`
- `outputs/our_work/pso_analysis/pso_supplement/tables/best_samples.csv`
- `outputs/our_work/pso_analysis/pso_supplement/figures/mse_by_target.png`
- `outputs/our_work/pso_analysis/pso_supplement/figures/accepted_count_heatmap.png`
- `outputs/our_work/pso_analysis/pso_supplement/figures/structures/material_frequency.png`
- `outputs/our_work/pso_analysis/pso_supplement/figures/spectra/<target_id>/layer_<layer_count>_topk.png`
- `outputs/our_work/pso_analysis/pso_supplement/figures/spectra/<target_id>/layer_<layer_count>_mean_band.png`
- `outputs/our_work/pso_analysis/pso_supplement/figures/lorentzian/center_vs_best_mse.png`

#### Step 4.5: 鐢熸垚 GA 浼樼瑙ｆ棌琛ュ厖鏁版嵁闆?

GA 琛ュ厖鏁版嵁闆嗙敤浜庝粠宸茬煡浼樼缁撴瀯鍑哄彂鍋氬眬閮ㄥ彉寮傚拰浜ゅ弶锛屾悳绱㈡弧瓒抽槇鍊肩殑鐩歌繎浼樼瑙ｆ棌銆傚綋鍓嶅彧鍖呭惈涓夌被鐩爣锛?

- `broad_3_13_high`锛歚3-13 um` 楂樺惛鏀讹紝鍏朵粬娉㈡涓嶅弬涓?loss銆傜瀛愮粨鏋勶細`YbF3(870) / ZnS(480) / Si(280) / Bi(20) / Ge(130) / Bi(820) / Au(100)`銆?
- `mid_5_8_high`锛歚3-5 um` 浣庡惛鏀躲€乣5-8 um` 楂樺惛鏀躲€乣8-13 um` 浣庡惛鏀讹紝鍏朵粬娉㈡涓嶅弬涓?loss銆傜瀛愮粨鏋勶細`Si(250) / SiO2(120) / Ge(500) / MgF2(850) / Ge(110) / MgF2(500) / Bi(130) / Au(100)`銆?
- `dual_3_5_8_13_high`锛歚3-5 um` 楂樺惛鏀躲€乣5-8 um` 浣庡惛鏀躲€乣8-13 um` 楂樺惛鏀讹紝鍏朵粬娉㈡涓嶅弬涓?loss銆傜瀛愮粨鏋勶細`SiO2(150) / MgF2(500) / Si(500) / ZnS(450) / Ge(490) / MgF2(280) / Si(320) / Bi(250) / Au(100)`銆?

鍗曡繘绋嬭繍琛岋細

```bash
cd /srv/OptoGPT-GRPO
python -m ga.scripts.run_ga_dataset --config ga/configs/ga_seeded_absorbers.yaml
```

GA 涓诲叆鍙ｇ幇鍦ㄧ洿鎺ユ敮鎸?`targets.tasks` 鑷畾涔変换鍔″垪琛細
- 榛樿閰嶇疆 [ga/configs/ga_seeded_absorbers.yaml](/O:/Optics%20Code/OptoGPT-GRPO/ga/configs/ga_seeded_absorbers.yaml) 宸叉樉寮忓啓鍏?3 涓?seeded 浠诲姟銆?
- 鑷畾涔変换鍔℃ā鏉胯 [ga/configs/ga_custom_tasks.yaml](/O:/Optics%20Code/OptoGPT-GRPO/ga/configs/ga_custom_tasks.yaml)銆?
- 姣忎釜浠诲姟鍙湪 `bands` 涓０鏄庨珮鍚告敹/浣庡惛鏀舵尝娈碉紱鏈０鏄庣殑娉㈡涓嶄細杩涘叆 loss銆?
- 鑻ユ彁渚?`seed_tokens`锛岃剼鏈細鍏堟鏌ュ帤搴﹀苟鎷嗗垎鍒濆 seed 涓?`>500 nm` 鐨勫眰锛涜嫢涓嶆彁渚涳紝鍒欐寜 `random_init` 鍦ㄥ悎娉曟潗鏂欏拰鍘氬害缃戞牸涓婇殢鏈虹敓鎴愬垵濮嬬粨鏋勩€?
- `targets.include_ids` 鍙敤浜庡彧杩愯浠诲姟鍒楄〃涓殑閮ㄥ垎 target銆?

褰撳墠 GA 閲囩敤鍥哄畾棰勭畻鎼滅储銆傛瘡涓?target 浼氳窇瀹?`search.restart_count * search.generations_per_restart`锛屼笉浼氬洜涓烘牱鏈暟閲忚揪鍒颁笂闄愬氨鎻愬墠鍋滄锛涘€欓€夋睜婊″悗锛屾柊鐨勬洿浼樻牱鏈細鏇挎崲褰撳墠杈冨樊鏍锋湰銆?

璇ユ楠ゅ畬鎴愬悗搴斿嚭鐜帮細

- `outputs/our_work/data_gen/ga_seeded_absorbers/shards/shard-00000.parquet`
- `outputs/our_work/data_gen/ga_seeded_absorbers/splits/split_manifest.json`
- `outputs/our_work/data_gen/ga_seeded_absorbers/vocab/vocab.json`
- `outputs/our_work/data_gen/ga_seeded_absorbers/targets/target_manifest.json`
- `outputs/our_work/data_gen/ga_seeded_absorbers/stats/summary.json`
- `outputs/our_work/data_gen/ga_seeded_absorbers/stats/search_summary.json`
- `outputs/our_work/data_gen/ga_seeded_absorbers/figures/*_accepted_absorption_topk.png`
- `outputs/our_work/data_gen/ga_seeded_absorbers/figures/*_mse_hist.png`

璇存槑锛?

- GA 鍥哄畾浣跨敤姣忎釜绉嶅瓙缁撴瀯鐨勫眰鏁帮紝鍦ㄥ悓灞傛暟鍐呭仛鏉愭枡鍙樺紓銆佸帤搴﹀彉寮傘€佺簿鑻变繚鐣欍€侀敠鏍囪禌閫夋嫨鍜?layer-wise crossover銆?
- 鎺ュ彈鏉′欢鏄?masked absorption MSE `< 0.005`锛屾瘡涓洰鏍囬粯璁ゆ敹闆?`100` 鏉″叏灞€鍘婚噸缁撴瀯銆?
- 榛樿鏉愭枡闆嗗悎涓?PSO 涓€鑷达紝浣跨敤 `database_dir` 涓嬬殑鍏ㄩ儴鏉愭枡锛涘鏋滃彧鎯冲洿缁曞凡鐭ヤ紭绉€瑙ｇ殑鏉愭枡灞€閮ㄦ悳绱紝鍙湪 YAML 閲屾樉寮忓啓 `materials`銆?
- 宸茬煡浼樼瑙ｅ寘鍚?`820/850/870 nm` 灞傦紱榛樿閰嶇疆浼氭妸杩欎簺 seed 鍘氬害棰濆鍔犲叆鍙€夊帤搴﹂泦鍚堛€傝嫢瑕佷弗鏍奸檺鍒跺埌 `10-500 nm`锛屽皢 `data.include_seed_thickness_values` 鏀逛负 `false`锛屼絾绉嶅瓙浼氳杩戦偦鍘氬害瑁佸壀锛屾悳绱㈣川閲忓彲鑳戒笅闄嶃€?
- 杈撳嚭鍏夎氨浠嶇劧鏄?`[R..., T...]`锛屽叡 `2048` 缁达紱鐩爣鍚告敹璋卞彧鐢ㄤ簬 GA 鎼滅储鏃惰绠?masked MSE銆?

澶氳繘绋嬫媶鍒嗚繍琛岋細

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=3 -m ga.scripts.run_ga_dataset --config ga/configs/ga_seeded_absorbers.yaml
```

浣跨敤澶氳繘绋嬪墠锛岄渶瑕佸厛鎶?`ga/configs/ga_seeded_absorbers.yaml` 閲岀殑 `distributed.enabled` 鏀规垚 `true`銆備笁涓?target 浼氭寜 rank 鎷嗗垎锛屽苟鍒嗗埆鍐欏埌锛?

- `outputs/our_work/data_gen/ga_seeded_absorbers/rank00`
- `outputs/our_work/data_gen/ga_seeded_absorbers/rank01`
- `outputs/our_work/data_gen/ga_seeded_absorbers/rank02`

濡傛灉鍙兂浠庢煇涓?parquet shard 閲岄殢鏈烘娊鏍风敾鍥撅紝渚嬪浠?`shard-00000.parquet` 闅忔満鎶?10 鏉?`3-13 um` 鐩爣鍏夎氨锛?

```bash
cd /srv/OptoGPT-GRPO
python -m ga.scripts.plot_random_parquet_spectra \
  --shard-path outputs/our_work/data_gen/ga_seeded_absorbers/shards/shard-00000.parquet \
  --output-path outputs/our_work/data_gen/ga_seeded_absorbers/figures/random_10_broad_3_13_absorption.png \
  --sample-count 10 \
  --seed 42 \
  --target-id broad_3_13_high
```

璇ュ懡浠や細鍚屾椂鍐欏嚭锛?

- `outputs/our_work/data_gen/ga_seeded_absorbers/figures/random_10_broad_3_13_absorption.png`
- `outputs/our_work/data_gen/ga_seeded_absorbers/figures/random_10_broad_3_13_absorption.selected.json`

#### Step 5: 鍚姩棰勮缁?

```bash
cd /srv/OptoGPT-GRPO
python pretrain/scripts/run_pretrain.py \
  --model-config pretrain/configs/model/base_gpt.yaml \
  --train-config pretrain/configs/train/base_train.yaml
```

鍏稿瀷缁堢杈撳嚭锛?

```text
{'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}
{'eval_loss': ..., 'eval_token_accuracy': ..., 'eval_runtime': ..., 'epoch': ...}
100%|鈻堚枅鈻堚枅鈻堚枅鈻堚枅鈻堚枅| ...
```

璇ユ楠ゅ畬鎴愬悗搴斿嚭鐜帮細

- `outputs/our_work/pretrain/base_train/checkpoint-*`
- `outputs/our_work/pretrain/base_train/checkpoint-*/config.json`
- `outputs/our_work/pretrain/base_train/checkpoint-*/model.safetensors`
- `outputs/our_work/pretrain/base_train/checkpoint-*/vocab.json`
- `outputs/our_work/pretrain/base_train/tensorboard/`
- `outputs/our_work/pretrain/base_train/metrics/train_metrics.jsonl`
- `outputs/our_work/pretrain/base_train/metrics/eval_metrics.jsonl`
- `outputs/our_work/pretrain/base_train/metrics/train_metrics.csv`
- `outputs/our_work/pretrain/base_train/metrics/eval_metrics.csv`
- `outputs/our_work/pretrain/base_train/plots/train_loss.png`
- `outputs/our_work/pretrain/base_train/plots/learning_rate.png`
- `outputs/our_work/pretrain/base_train/plots/grad_norm.png`
- `outputs/our_work/pretrain/base_train/plots/eval_loss.png`
- `outputs/our_work/pretrain/base_train/plots/eval_token_accuracy.png`
- `outputs/our_work/pretrain/base_train/plots/overview.png`

浣犲彲浠ユ鏌ワ細

```bash
ls outputs/our_work/pretrain/base_train
ls outputs/our_work/pretrain/base_train/checkpoint-1
```

TensorBoard 瀹炴椂鏌ョ湅鍛戒护锛?

```bash
cd /srv/OptoGPT-GRPO
tensorboard --logdir outputs/our_work/pretrain/base_train/tensorboard --bind_all
```

4 鍗?A100 姝ｅ紡鍛戒护锛?

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=4 pretrain/scripts/run_pretrain.py \
  --model-config pretrain/configs/model/base_gpt.yaml \
  --train-config pretrain/configs/train/a100_4gpu.yaml
```

4 鍗￠粯璁よ缁冮厤缃鐐癸細

- 璇诲彇鏁版嵁闆嗭細`outputs/our_work/data_gen/a100_4gpu`
- `per_device_train_batch_size: 512`
- `per_device_eval_batch_size: 512`
- `num_train_epochs: 100`
- `learning_rate: 1e-4`
- `lr_scheduler_type: cosine`
- `warmup_ratio: 0.01`
- `max_grad_norm: 1.0`
- `logging_steps: 1000`
- `eval_steps: 5000`
- `save_steps: 5000`
- `save_total_limit: 3`

8 鍗?A100 姝ｅ紡鍛戒护锛?

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=8 pretrain/scripts/run_pretrain.py \
  --model-config pretrain/configs/model/base_gpt.yaml \
  --train-config pretrain/configs/train/a100_8gpu.yaml
```

#### Step 6: 杩愯鐙珛璇勬祴

```bash
cd /srv/OptoGPT-GRPO
python pretrain/scripts/run_eval.py \
  --checkpoint-dir outputs/our_work/pretrain/base_train \
  --dataset-dir outputs/our_work/data_gen/v1 \
  --database-dir database \
  --split val \
  --max-samples 256 \
  --max-new-tokens 10 \
  --output-dir outputs/our_work/eval \
  --output-json outputs/our_work/eval/latest_eval.json
```

鍏稿瀷缁堢杈撳嚭锛?

```text
our_work eval:  42%|鈻堚枅鈻堚枅鈻?    | 108/256 [00:xx<00:xx, ... sample/s, valid=..., exact=..., last_rmse=...]
{
  "summary": {
    "sample_count": 256,
    "valid_generation_count": ...,
    ...
  },
  "results": [...],
  "run_dir": ".../outputs/our_work/eval/base_train/eval_runs/2026..."
}
```

璇ユ楠ゅ畬鎴愬悗搴斿嚭鐜帮細

- `outputs/our_work/eval/base_train/eval_runs/<timestamp>/summary.json`
- `outputs/our_work/eval/base_train/eval_runs/<timestamp>/results.jsonl`
- `outputs/our_work/eval/latest_eval.json`

濡傛灉涓嶅姞 `--disable-plots`锛岃繕浼氬嚭鐜帮細

- `outputs/our_work/eval/base_train/eval_runs/<timestamp>/plots/*.png`
- `outputs/our_work/eval/base_train/eval_runs/<timestamp>/samples/*.png`

#### Step 7: 杩愯 our_work 杞婚噺 GRPO

`rl` 褰撳墠鏄竴涓交閲忋€佽缁冨氨缁殑 GRPO 瀛愮郴缁燂紝鎺ュ彛椋庢牸灏介噺璐磋繎 `Transformers + torchrun`锛屼絾娌℃湁寮曞叆閲嶅瀷澶栭儴 RL 骞冲彴銆?

鍗曟満鍗曞崱 smoke锛?

```bash
cd /srv/OptoGPT-GRPO
python rl/scripts/run_grpo.py --config rl/configs/grpo/base_grpo.yaml
```

鍩虹 RL 閰嶇疆瑕佺偣锛?

- `model.checkpoint_dir: outputs/our_work/pretrain/base_train`
- `data.dataset_dir: outputs/our_work/data_gen/v1`
- `per_device_batch_size: 16`
- `rollout.batch_size: 512`
- `scoring.batch_size: 1024`
- `reward.tmm.batch_size: 4096`
- `training.lr_scheduler_type: cosine`
- `training.warmup_ratio: 0.01`

4 鍗?A100 姝ｅ紡鍛戒护锛?

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=4 rl/scripts/run_grpo.py --config rl/configs/grpo/a100_4gpu.yaml
```

4 鍗?RL 閰嶇疆瑕佺偣锛?

- `model.checkpoint_dir: outputs/our_work/pretrain/a100_4gpu`
- `data.dataset_dir: outputs/our_work/data_gen/a100_4gpu`
- `data.num_workers: 0`
- `per_device_batch_size: 32`
- `gradient_accumulation_steps: 1`
- `rollout.batch_size: 128`
- `scoring.batch_size: 256`
- `reward.tmm.batch_size: 128`
- `training.lr_scheduler_type: cosine`
- `training.warmup_ratio: 0.01`
- `training.eval_steps: 1000`
- `training.save_steps: 1000`

8 鍗?A100 姝ｅ紡鍛戒护锛?

```bash
cd /srv/OptoGPT-GRPO
torchrun --nproc_per_node=8 rl/scripts/run_grpo.py --config rl/configs/grpo/a100_8gpu.yaml
```

8 鍗?RL 閰嶇疆瑕佺偣锛?

- `model.checkpoint_dir: outputs/our_work/pretrain/a100_8gpu`
- `data.dataset_dir: outputs/our_work/data_gen/a100_8gpu`
- `data.num_workers: 0`
- `per_device_batch_size: 32`
- `gradient_accumulation_steps: 1`
- `rollout.batch_size: 128`
- `scoring.batch_size: 256`
- `reward.tmm.batch_size: 128`
- `training.lr_scheduler_type: cosine`
- `training.warmup_ratio: 0.01`
- `training.eval_steps: 1000`
- `training.save_steps: 1000`

鍏稿瀷缁堢杈撳嚭锛?

```text
our_work grpo: 100%|鈻堚枅鈻堚枅鈻堚枅鈻堚枅鈻堚枅| ... [loss=..., reward=..., valid=...]
```

璇ユ楠ゅ畬鎴愬悗搴斿嚭鐜帮細

- `outputs/our_work/rl/<run-name>/metrics/train_metrics.jsonl`
- `outputs/our_work/rl/<run-name>/metrics/eval_metrics.jsonl`
- `outputs/our_work/rl/<run-name>/metrics/train_metrics.csv`
- `outputs/our_work/rl/<run-name>/metrics/eval_metrics.csv`
- `outputs/our_work/rl/<run-name>/plots/train_loss.png`
- `outputs/our_work/rl/<run-name>/plots/train_mean_reward.png`
- `outputs/our_work/rl/<run-name>/plots/train_valid_ratio.png`
- `outputs/our_work/rl/<run-name>/plots/eval_mean_reward.png`
- `outputs/our_work/rl/<run-name>/plots/overview.png`
- `outputs/our_work/rl/<run-name>/tensorboard/`
- `outputs/our_work/rl/<run-name>/checkpoints/checkpoint-*`

濡傞渶瀹炴椂鏌ョ湅 RL 鏍囬噺锛屽彲鎵ц锛?

```bash
tensorboard --logdir outputs/our_work/rl/<run-name>/tensorboard --bind_all
```

### 5. 鍙儴缃插凡鏈?checkpoint 鍋氳瘎娴?
濡傛灉浣犱笉鎯冲湪鏈嶅姟鍣ㄤ笂閲嶈锛屽彧鎯宠瘎娴嬪凡鏈夋ā鍨嬶紝闇€瑕佸悓姝ワ細

- 浠撳簱浠ｇ爜
- `database/`
- 宸茬敓鎴愮殑鏁版嵁闆嗙洰褰曪紝渚嬪 `outputs/our_work/data_gen/v1`
- 宸茶缁冪殑 checkpoint 鐩綍锛屼緥濡?`outputs/our_work/pretrain/base_train`

鐒跺悗鐩存帴杩愯锛?

```bash
cd /srv/OptoGPT-GRPO
python pretrain/scripts/run_eval.py \
  --checkpoint-dir outputs/our_work/pretrain/base_train \
  --dataset-dir outputs/our_work/data_gen/v1 \
  --database-dir database \
  --split val \
  --max-samples 256 \
  --max-new-tokens 10 \
  --output-dir outputs/our_work/eval \
  --output-json outputs/our_work/eval/latest_eval.json
```

### 6. 甯歌闂
- `database_path must point to an existing directory`
  - 鍘熷洜锛氭湇鍔″櫒娌″悓姝?`database/`
  - 妫€鏌ワ細`ls database`
- `No checkpoint directory found under ...`
  - 鍘熷洜锛氳缁冪洰褰曢噷娌℃湁 `checkpoint-*`
  - 妫€鏌ワ細`ls outputs/our_work/pretrain/base_train`
- `spectrum_dim mismatch`
  - 鍘熷洜锛歚base_gpt.yaml` 閲岀殑 `model.spectrum_dim` 涓庢暟鎹泦 `2 * num_points` 涓嶄竴鑷?
- `num_points mismatch`
  - 鍘熷洜锛歚run_eval.py` 鍛戒护琛屼紶鍏ョ殑 `--num-points` 涓?checkpoint 鐨?`spectrum_dim` 涓嶄竴鑷?
- `read_excel` / parquet 鐩稿叧鎶ラ敊
  - 鍘熷洜锛氱己灏?`openpyxl` 鎴?`pyarrow`

## 璇存槑
- 褰撳墠 `physics/` 鐩存帴澶嶇敤鍘?TMM 妯″潡锛屼笉鍙﹁捣涓€濂楀疄鐜般€?
- 褰撳墠璁粌鐩爣涓嶆槸浼犵粺 teacher forcing CE锛岃€屾槸鍩轰簬鐩爣鍏夎氨 reward 鐨?GRPO銆?
- rollout 涓?update 鐜板湪鍏辩敤鍚屼竴 policy 瀹氫箟锛屼笉鍐嶅嚭鐜?鈥渇iltered rollout / raw scoring鈥?鐨勪笉涓€鑷淬€?
- 褰撳墠榛樿鐨勫厜璋辫宸槸 `R/T` 鐩存帴璇樊锛屽嵆姣旇緝鎷兼帴鍚庣殑 `[R..., T...]` 鍏夎氨銆?
- `core/` 淇濈暀鐨勪富瑕佺洰鐨勶紝鏄吋瀹规棫 OptoGPT checkpoint 鐨勫姞杞姐€?
## our_work Eval Suite

鐢ㄩ€旓細

- 鍔犺浇 `pretrain` 璁粌濂界殑 checkpoint
- 鍚屾椂璇勪及 `train + val`
- 鍚?split 闅忔満鎶芥牱鍥哄畾鏁伴噺鏍锋湰
- 鎵归噺鐢熸垚棰勬祴缁撴瀯
- 鎵归噺鍥炵畻棰勬祴缁撴瀯鍏夎氨
- 璁＄畻鐩爣鍏夎氨涓庨娴嬪厜璋辫宸?
- 杈撳嚭姹囨€?JSON / JSONL
- 杈撳嚭鏈€濂?/ 鏈€宸?/ 鎺ヨ繎鍧囧€艰宸牱鏈殑搴忓垪涓庡厜璋卞姣斿浘

杩愯鏂瑰紡锛?

```bash
python eval/scripts/run_eval_suite.py --config eval/configs/base_eval.yaml
```

榛樿閰嶇疆鏂囦欢锛?

- `eval/configs/base_eval.yaml`

涓昏閰嶇疆椤癸細

- `paths.checkpoint_dir`
- `paths.dataset_dir`
- `paths.database_dir`
- `paths.output_dir`
- `data.splits`
- `data.sample_mode`
- `data.max_samples_per_split`
- `data.max_shards_per_split`
- `inference.batch_size`
- `inference.max_new_tokens`
- `tmm.batch_size`
- `plots.worst_count`
- `plots.best_count`
- `plots.mean_count`

閲囨牱妯″紡锛?

- `random`
  - 鎵弿鏁翠釜 split 鐨勬墍鏈?shard锛岀敤 reservoir sampling 鍋氫弗鏍奸殢鏈烘娊鏍?
- `head_shards`
  - 鍙壂鎻忓墠鑻ュ共涓?shard锛岄€熷害鏈€蹇紝浣嗘牱鏈彲鑳芥湁椤哄簭鍋忓樊
- `shard_subset_random`
  - 鍏堥殢鏈洪€夎嫢骞蹭釜 shard锛屽啀鍙壂鎻忚繖浜?shard锛岄€熷害鍜屼唬琛ㄦ€ф姌涓?

杈撳嚭鍐呭锛?

- `summary.json`
- `split_summaries.json`
- `selected_samples.json`
- `results/train.jsonl`
- `results/val.jsonl`
- `plots/train/*.png`
- `plots/val/*.png`
- `plots/comparison/*.png`
- `samples/train/best/*.png`
- `samples/train/worst/*.png`
- `samples/train/mean/*.png`
- `samples/val/best/*.png`
- `samples/val/worst/*.png`
- `samples/val/mean/*.png`


