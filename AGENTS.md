# AGENTS

## 环境
- Conda 环境名：待确认。
- 运行前先确认 `python`、`torch`、`PyYAML`、`pandas`、`scipy` 可用；需要保存图像时再确认 `matplotlib`。

## 修改前先读
- 先读 `README.md`、`docs/notes.md`、当前需求涉及的代码文件和配置文件。
- 涉及训练入口时，至少补读 `runners/run_grpo.py` 和 `configs/grpo/spectral_grpo.yaml`。
- 涉及评测入口时，至少补读 `runners/run_spectrum_eval.py` 和 `configs/eval/spectrum_eval.yaml`。
- 涉及采样、光谱损失、TMM 或 checkpoint 时，补读对应模块再改。

## 禁止乱改
- 没有明确需求时，不改业务逻辑、算法流程、配置参数值、目录结构、输出文件名或 checkpoint 兼容接口。
- 不要顺手删除 `core/`、`core/trains/`、`physics/` 等兼容或底层模块。
- 信息不足就写“待确认”，不要补写未经仓库验证的路径、环境名或运行结论。

## 修改规范
- 只改与当前需求直接相关的文件，保持最小改动。
- 改 Python 时补必要注释，尤其是非直观分支、张量形状假设和兼容性代码。
- 改 YAML 时，新增项或重要修改必须加注释，说明用途或约束。
- 改路径、输出或命令前，先在仓库内核对真实文件和调用链。

## 文档更新规则
- 每次实际修改后都要更新 `docs/notes.md` 和 `docs/logs/当前月份.md`。
- `docs/notes.md` 只保留最近一次修改摘要，直接覆盖。
- `docs/logs/当前月份.md` 追加当次记录，不改旧记录语义。

## Git 规范
- 分支名建议使用 `type/short-topic`，如 `docs/doc-system-shrink`。
- commit message 建议使用 `type: summary`，如 `docs: consolidate project documentation`。

## 修改后输出格式
- 列出实际修改、创建、删除的文件。
- 简述每个改动文件的作用或主要变化。
- 写明验证命令与结果；未验证就直接写“未验证”。
- 给出建议的 branch 名称和 commit message。
