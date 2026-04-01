# Dynamic Target Attack

[English README](README_EN.md)

这个仓库是 **Dynamic Target Attack (DTA)** 项目的一个精简脱敏版本，只保留了核心攻击流程、消融实验入口，以及可复用的 harmfulness 评测工具链，便于整理、协作和对外展示。

## 当前版本特点

- 去掉了个人机器绝对路径
- API 配置改为环境变量读取
- 统一入口放在 `src/main.py`
- 删除了大量历史实现、一次性脚本和外围评测代码

## 仓库定位

这个版本有意只保留最核心的研究链路，不再包含：

- 旧版 attacker 实现
- 一次性分析脚本
- 基线方法专用评测辅助脚本
- 与当前主流程无直接关系的外围文件

当前主要维护的代码路径是：

- `src/combined_attacker.py`：组合式 DTA 主流程
- `src/experiments_ablation.py`：消融实验入口
- `src/eval/harmfulness/`：可复用的 harmfulness 评测模块
- `src/attacker_v3.py`：保留下来的主攻击实现

## 目录结构

```text
.
├── README.md
├── README_EN.md
├── README_zh.md
├── .env.example
├── requirements.txt
├── pyproject.toml
├── environment.yaml
├── data/
│   ├── README.md
│   └── raw/
├── scripts/
│   ├── run_ablation_A1.sh
│   ├── run_ablation_F_batch.sh
│   └── run_combined_attacker01.sh
└── src/
    ├── main.py
    ├── combined_attacker.py
    ├── experiments_ablation.py
    ├── attacker_v3.py
    └── eval/harmfulness/
```

## 主要入口

如果你希望从统一入口使用：

```bash
python src/main.py combined --help
python src/main.py ablation --help
python src/main.py harmfulness-eval --help
```

也可以直接调用底层模块：

```bash
python src/combined_attacker.py --help
python src/experiments_ablation.py --help
python src/eval/run_harmfulness_eval.py --help
```

## 安装方式

### 方式 1：uv

```bash
uv sync
```

### 方式 2：pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 方式 3：conda

```bash
conda env create -f environment.yaml
conda activate dta
```

## 环境变量

先复制模板文件：

```bash
cp .env.example .env
```

常见变量包括：

- `OPENAI_API_KEY`：OpenAI 兼容后端所需
- `OPENAI_API_BASE`：可选，自定义兼容接口地址
- `ANTHROPIC_API_KEY`、`GOOGLE_API_KEY`、`DASHSCOPE_API_KEY`、`TOGETHER_API_KEY`：按需配置
- `DTA_DATA_DIR`：覆盖默认数据目录 `./data`
- `DTA_LOG_DIR`：覆盖默认日志目录 `./logs`
- `DTA_MODEL_LLAMA3`、`DTA_MODEL_QWEN25`、`DTA_GPTFUZZ_MODEL` 等：如果你使用本地镜像或私有模型路径，可以在这里覆盖默认值

## 数据目录约定

默认情况下，项目假设数据放在 `data/` 下：

```text
data/
  raw/
    advbench_100.csv
    harmBench_100.csv
```

CSV 至少需要包含以下任一列：

- `goal`
- `harmful`

常见输出目录包括：

- `data/combined/`
- `data/DTA_ablation/`
- `data/eval/`
- `logs/`

详细说明见 [data/README.md](data/README.md)。

## 常用工作流

### 1. 运行组合攻击主流程

```bash
python src/main.py combined --target-llm Llama3 --help
```

保留下来的示例批处理脚本：

```bash
bash scripts/run_combined_attacker01.sh
```

### 2. 运行消融实验

```bash
python src/main.py ablation --experiment A1 --help
```

示例批处理脚本：

```bash
bash scripts/run_ablation_A1.sh
```

### 3. 运行 harmfulness 评测

```bash
python src/main.py harmfulness-eval --input-path path/to/results.jsonl --help
```

## 快速自检

安装依赖后，建议先检查这些命令：

```bash
python src/main.py --help
python src/main.py combined --help
python src/main.py ablation --help
python src/main.py harmfulness-eval --help
```

## 使用说明

这个仓库主要用于安全研究、红队测试、防御评估和可复现性实验，不包含原始数据集、模型权重或私有实验结果。运行前请遵守所在机构规范、模型服务条款和相关法律要求。
