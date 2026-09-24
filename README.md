# Semantic

重构后的 LLM 文本统计物理 / TWO-NN / FSS 项目。

## 模型和数据放哪里

**模型不要放进 `src/`。** 默认放在项目根目录的 `models/`：

```text
Semantic/models/Qwen/Qwen3-32B/
```

如果模型已经在集群公共目录（推荐），不要复制几十 GB 权重；直接在 `configs/qwen32b.yaml` 把 `model.path` 改成绝对路径，例如：

```yaml
model:
  path: /public/home/.../Qwen3-32B
```

**原始数据**放 `data/raw/`，只做保存、不覆盖；**清洗/筛选后、真正用于实验的数据**放 `data/processed/`。例如旧项目的 `./FSS/filter_v0.json` 建议移动为：

```text
Semantic/data/processed/filter_v0.json
```

大型数据如果已有固定目录，同样可以不复制，直接把 experiment YAML 的 `experiment.input` 写成绝对路径。

**输出和缓存**统一进入 `outputs/<experiment-name>/`。contextual 模型计算得到的向量 checkpoint 会自动放在该实验的 `vectors/checkpoints/`，程序中断后重新运行会复用已经完成的 `(temperature, size)`。

## 推荐目录语义

- `configs/`: 参数，不放算法。
- `src/semantic/`: 可复用的科学计算代码。
- `scripts/`: 很薄的命令行入口。
- `models/`: 本地模型权重；默认 git ignore。
- `data/raw/`: 原始生成数据；默认 git ignore。
- `data/processed/`: 清洗、截断、筛选后的实验输入；默认 git ignore。
- `outputs/`: 每次实验的结果、图、checkpoint；默认 git ignore。
- `prompts/`: prompt 文本。
- `tests/`: 数值回归和缓存测试。

## 安装

```bash
cd Semantic
pip install -e .
```

## 运行 static FSS

```bash
python scripts/run_fss.py --config configs/experiments/fss_static.yaml
```

## 运行 contextual FSS

```bash
python scripts/run_fss.py --config configs/experiments/fss_contextual.yaml
```

不再需要维护两份 FSS 主程序；static/contextual 的差别只由 `representation.type` 决定。

## 配置覆盖顺序

`default.yaml -> qwen32b.yaml -> experiment.yaml`，后面的值覆盖前面的值。因此模型地址只改模型配置，实验参数只改 experiment 配置。

## 从旧项目迁移

1. `filter_v0.json` -> `data/processed/filter_v0.json`。
2. Qwen 权重 -> `models/Qwen/Qwen3-32B/`，或者把 YAML `model.path` 指向原来的绝对路径。
3. prompt -> `prompts/*.txt`。
4. 旧 `FSS/`, `FSS_enc/`, `analysis/` 结果先保留，不要覆盖；新结果全部写 `outputs/`。
5. 确认新旧 TWO-NN/FSS 数值一致后，再删除旧脚本。

## 一个重要设计

`representations/` 只负责 `text -> vector`；`metrics/two_nn.py` 只负责 `vectors -> intrinsic dimension`；`fss/` 只负责有限尺寸标度；plotting 不负责计算。这样以后新增 layer sweep、不同 pooling、whitening、correlation observable 时，不需要复制整个 pipeline。
