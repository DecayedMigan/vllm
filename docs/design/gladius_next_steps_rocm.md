# GLADIUS vLLM：P1 收口与 ROCm GPU 验证

本清单是 `feature/gladius-v3-vllm-engine` 在 canonical contract 1.0.0
冻结后的下一步执行目标。执行主机为 `100.72.102.42`，仓库为：

```text
/home/zazzi/CODE/vllm-src-gladius
```

## 强制环境

所有 Python、pytest 和 vLLM 命令必须使用现有 `llm-finetune` 环境：

```bash
export GLADIUS_VLLM_ROOT=/home/zazzi/CODE/vllm-src-gladius
export GLADIUS_PYTHON=/home/zazzi/miniconda3/envs/llm-finetune/bin/python
cd "$GLADIUS_VLLM_ROOT"

"$GLADIUS_PYTHON" -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))'
```

已核实该环境为 Python 3.12、PyTorch `2.12.0+rocm7.2`，能够识别
`Radeon RX 7900 XTX`。禁止使用系统 `/usr/bin/python3`。

当前环境中的 `vllm` 指向 `/home/zazzi/CODE/vllm-src`，不是本 feature
仓库。每次测试前必须检查：

```bash
PYTHONPATH="$GLADIUS_VLLM_ROOT" "$GLADIUS_PYTHON" -c \
  'import vllm; print(vllm.__file__)'
```

纯 Python GLADIUS 测试可通过 `PYTHONPATH` 直接使用本仓库。真实 engine
测试若加载到旧的扩展或旧源码，使用 `uv` 将当前仓库安装进指定环境：

```bash
/home/zazzi/.local/bin/uv pip install \
  --python "$GLADIUS_PYTHON" \
  -e "$GLADIUS_VLLM_ROOT" \
  --torch-backend=auto
```

不要使用裸 `pip`，也不要修改 `/home/zazzi/CODE/vllm-src` 中已有的依赖改动。

## P1-A：关闭四个 contract/fail-safe 缺口

严格执行测试先行，每项先观察失败，再实现最小修复。

1. **严格 SemVer**
   - `parse_policy_snapshot()` 只接受 `MAJOR.MINOR.PATCH` 数字格式。
   - reader 接受兼容 major 1；publisher fixture 固定 `1.0.0`。
   - 必须拒绝 `1`、`1.0`、`1.foo`、整数 `1` 和 major 2。

2. **`stat()` never-raises**
   - `PolicyLoader.poll()` 的 `Path.stat()` 捕获所有 `OSError`。
   - `PermissionError`、临时 I/O 错误不得逃出 scheduler。
   - corrupt/mismatch/regression 仅保留尚未过期的 last-good。

3. **生产 polling interval 禁止 0**
   - 环境变量 `GLADIUS_POLICY_POLL_INTERVAL_MS` 最小值为 1。
   - 测试需要即时 polling 时，通过 `PolicyLoader(..., poll_interval_ms=0)`
     显式注入，并在代码中标记为测试用途。

4. **Telemetry rotation**
   - 支持按文件大小轮转，且不得拆分 JSONL 记录。
   - rotate/open/write/flush/close 的任意失败均 fail-open。
   - serving 热路径不得因 telemetry 异常失败或重复刷警告。

## P1-B：CPU 与 contract 验证

先运行无 GPU 的 GLADIUS 测试：

```bash
cd "$GLADIUS_VLLM_ROOT"
PYTHONPATH="$GLADIUS_VLLM_ROOT" "$GLADIUS_PYTHON" -m pytest \
  --confcutdir=tests/gladius \
  tests/gladius/test_schema.py \
  tests/gladius/test_policy_loader.py \
  tests/gladius/test_telemetry_writer.py \
  -q
```

随后运行 CPU scheduler 测试：

```bash
PYTHONPATH="$GLADIUS_VLLM_ROOT" "$GLADIUS_PYTHON" -m pytest \
  tests/gladius/test_gladius_scheduler_cpu.py -q
```

必须覆盖：startup 等价、双 ceiling clamp、已有请求不驱逐、自然收敛、
missing/expired default、未过期 last-good，以及 telemetry fail-open。

共享 fixture 为：

```text
tests/gladius/fixtures/gladius-control-protocol-v1.json
```

它必须能被 SMIG 的 `PolicySnapshot.from_dict()` 和 vLLM 的
`parse_policy_snapshot()` 同时消费。禁止恢复已删除的旧
`vllm/v1/core/sched/gladius_*` 状态机。

## P1-C：RX 7900 XTX ROCm smoke

先记录环境证据：

```bash
rocm-smi --showproductname --showmeminfo vram --showuse --showtemp
"$GLADIUS_PYTHON" -c \
  'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

使用小模型完成机制验证，避免第一轮直接加载 8B。启动时必须显式指定：

```bash
export GLADIUS_ENGINE_ID=rx7900xtx-engine-a
export GLADIUS_POLICY_DIR=/tmp/gladius-policy
export GLADIUS_POLICY_POLL_INTERVAL_MS=100
export GLADIUS_TELEMETRY_SAMPLE_N=1
mkdir -p "$GLADIUS_POLICY_DIR"
```

建议 smoke 顺序：

1. 以 startup ceiling `max_num_seqs=16`、`max_num_batched_tokens=4096` 启动。
2. 发布 generation 1：`8 / 2048`。
3. 发布 generation 2：`2 / 512`。
4. 保持并发请求运行，确认降低 ceiling 不驱逐已运行请求。
5. 等待策略过期，确认恢复 startup default。
6. 检查 `telemetry.jsonl` 中 generation、policy、engine、model、step、
   requested/effective ceiling 和 clamped 完整对应。

## 完成门槛

只有同时满足以下条件，才可将 vLLM P1 标记完成：

- 四个 fail-safe 缺口都有失败测试和通过证据。
- Ruff、纯 contract、policy、telemetry 和 CPU scheduler 测试通过。
- RX 7900 XTX real-engine smoke 通过。
- SMIG 能无损读取真实 telemetry，并严格关联同一
  `(engine_id, model_id, generation, policy_id)`。
- 未恢复旧控制协议、旧环境变量或第二套有状态 scheduler。

真实 GPU/SLO 或跨仓契约未通过时，不得宣称持续学习 serving 闭环完成。
