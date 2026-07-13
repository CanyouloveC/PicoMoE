# PicoMoE

> 小内存，大模型 —— 基于 madvise 专家分页的 MoE 推理引擎

**PicoMoE** 让 DeepSeek V4 Flash (81GB) 这样的巨型 MoE 模型能在 **32GB 内存** 的普通机器上运行。核心思路是"把内存当显存用，把磁盘当内存用"——通过 madvise 系统调用精确控制哪些专家权重留在内存中，哪些可以被换出。

- 🌐 [English version](#picomoe-1)

---

## 目录

- [快速开始](#快速开始)
- [核心原理](#核心原理)
- [功能特性](#功能特性)
- [完整参数说明](#完整参数说明)
- [项目结构](#项目结构)
- [系统要求](#系统要求)
- [局限性](#局限性)
- [路线图](#路线图)
- [致谢](#致谢)

---

## 快速开始

### 编译

在 Linux 或 WSL2 上：

```bash
make cpu
```

### 下载模型

```bash
./download_model.sh q2-imatrix    # 约 81GB
```

### 运行

```bash
# 交互对话
./ds4 -m ds4flash.gguf --cpu --ssd-streaming -c 4096 --nothink

# 单次提问
./ds4 -m ds4flash.gguf --cpu --ssd-streaming -p "你好" --nothink -c 4096

# HTTP API 服务器（可对接 ChatBox、Open WebUI）
./ds4-server -m ds4flash.gguf --cpu --ssd-streaming -c 4096 --host 0.0.0.0 --port 8000
```

### 关键参数

| 参数 | 说明 |
|------|------|
| `--cpu` | 强制 CPU 后端 |
| `--ssd-streaming` | 启用专家分页（**必须开启**） |
| `-c, --ctx N` | 上下文窗口，32GB 内存建议 `4096` |
| `--nothink` | 关闭思考模式，加速响应 |
| `--cpu-usage-file PATH` | 专家使用量缓存文件 |

---

## 核心原理

```
┌──────────────────────────────────────────────────────┐
│  81GB GGUF 文件 (mmap 映射)                           │
│  ┌────────────┬─────────────────────────────────────┐│
│  │ 常驻部分    │         MoE 专家权重 (主体)          ││
│  │ ~10GB      │  E0 │ E1 │ E2 │ ... │ E255        ││
│  │            │  按需分页，动态管理                   ││
│  └────────────┴─────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
         │                      │
         ▼                      ▼
   始终在内存中           madvise 精确控制
                         ┌──────────────────────┐
                         │ WILLNEED   按需预取   │
                         │ DONTNEED   冷页淘汰   │
                         │ route-ahead 预测预取 │
                         │ 使用量学习  热启动   │
                         └──────────────────────┘
```

三个关键机制：

1. **按需预取**：路由选定专家后，立即用 `madvise(MADV_WILLNEED)` 通知内核异步预读
2. **LRU 淘汰**：用 `madvise(MADV_DONTNEED)` 释放长期未用的冷专家页面
3. **预测预取**：利用专家共现矩阵，在处理当前层时预测并预取下一层的专家

---

## 功能特性

### 模型支持

| 模型 | 大小 | 支持 |
|------|------|------|
| DeepSeek V4 Flash | 81GB (q2) | ✅ 完整 |
| DeepSeek V4 Pro | 更大 | ⚠️ 需更多内存 |

### 推理特性

| 特性 | 状态 | 说明 |
|------|------|------|
| CSA 压缩注意力 | ✅ 自动生效 | 核心架构，无需配置 |
| MTP 推测解码 | ⚠️ 实验性 | 提速有限，CPU 路径不推荐 |
| 1M 上下文窗口 | ❌ 内存受限 | 32GB 建议 4096 tokens |
| SSD 流式加载 | ✅ | 本项目的核心贡献 |
| 分布式推理 | ✅ | `--role coordinator/worker` |

### 后端支持

| 后端 | 状态 | 说明 |
|------|------|------|
| **CPU + SSD streaming** | ✅ **主力** | 32GB+ 内存即可 |
| Metal (macOS) | ✅ | 需 96GB+ 统一内存 |
| CUDA (NVIDIA) | ✅ | 需 24GB+ 显存 |
| ROCm (AMD) | ✅ | Strix Halo 等 |

### 内置工具

| 工具 | 用途 |
|------|------|
| `ds4` | 交互对话 / 单次提问 |
| `ds4-server` | OpenAI 兼容 HTTP API |
| `ds4-agent` | 终端编码 Agent（工具调用） |
| `ds4-bench` | 性能基准测试 |
| `ds4-eval` | 内置评测集 |

---

## 完整参数说明

### `ds4` — 交互对话

```
Usage: ds4 [(-p PROMPT | --prompt-file FILE)] [options]
```

**模型与运行时**

| 参数 | 说明 |
|------|------|
| `-m, --model FILE` | GGUF 模型路径，默认 `ds4flash.gguf` |
| `--cpu` / `--cuda` / `--metal` | 选择后端 |
| `--backend NAME` | 后端名称：`cpu` / `cuda` / `metal` |
| `-c, --ctx N` | 上下文窗口大小 |
| `-t, --threads N` | CPU 辅助线程数 |
| `--ssd-streaming` | 启用专家分页 |
| `--ssd-streaming-cold` | 跳过热门专家预加载 |
| `--ssd-streaming-cache-experts N\|NGB` | 缓存预算，CPU 默认自动检测 |
| `--cpu-usage-file PATH` | 使用量统计文件路径 |

**采样参数**

| 参数 | 说明 |
|------|------|
| `-p, --prompt TEXT` | 单次提问 |
| `--prompt-file FILE` | 从文件读取提问 |
| `-n, --tokens N` | 最大生成 token 数 |
| `--temp F` | 温度，0 为贪心 |
| `--top-p F` | 核采样 |
| `--seed N` | 随机种子 |
| `--think` / `--think-max` / `--nothink` | 推理模式 |

**交互命令**

| 命令 | 说明 |
|------|------|
| `/help` | 查看帮助 |
| `/think` `/nothink` | 切换思考模式 |
| `/ctx N` | 调整上下文并重启会话 |
| `/read FILE` | 读取文件作为消息 |
| `/quit` `/exit` | 退出 |
| `Ctrl+C` | 停止当前生成 |

### `ds4-server` — HTTP API

```
Usage: ds4-server [options]
```

| 参数 | 说明 |
|------|------|
| `--host HOST` | 绑定地址，默认 `127.0.0.1` |
| `--port N` | 端口，默认 `8000` |
| `--cors` | 允许跨域 |
| `-n, --tokens N` | 默认最大输出 |
| `--kv-disk-dir DIR` | 磁盘 KV 缓存 |
| `--kv-disk-space-mb N` | 缓存预算 |

端点：`/v1/chat/completions`、`/v1/responses`、`/v1/completions`、`/v1/messages`

### `ds4-agent` — 编码 Agent

```
Usage: ds4-agent [options]
```

| 参数 | 说明 |
|------|------|
| `-p, --prompt TEXT` | 初始任务 |
| `--non-interactive` | 非交互模式 |
| `--trace FILE` | 调试日志 |
| `--chdir DIR` | 工作目录 |

Agent 命令：`/save` `/list` `/switch` `/new` `/quit`

### `ds4-bench` — 性能测试

```
Usage: ds4-bench (--prompt-file FILE | --chat-prompt-file FILE) [options]
```

| 参数 | 说明 |
|------|------|
| `--ctx-start N` | 起始测量点 |
| `--ctx-max N` | 最大测量点 |
| `--gen-tokens N` | 每次解码 token 数 |
| `--csv FILE` | 输出 CSV |

### `ds4-eval` — 评测

```
Usage: ds4-eval [options]
```

| 参数 | 说明 |
|------|------|
| `--questions N` | 只跑前 N 题 |
| `--plain` | 禁用分屏 UI |
| `--trace FILE` | 评测日志 |
| `--regrade-trace FILE` | 重评分已有日志 |

---

## 项目结构

```
新增文件:
  ds4_cpu_stream.h         公开 API（~135 行）
  ds4_cpu_stream.c         核心实现（~450 行）
  CPU_STREAMING.md         技术文档

修改文件:
  ds4.c                    桥接层 + 解码路径集成
  Makefile                 编译目标
  ds4_cli.c                --cpu-usage-file 参数
  ds4_help.c               帮助文本更新
```

---

## 系统要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| 内存 | 32GB | 64GB+ |
| 磁盘 | SSD + 100GB 可用 | NVMe |
| 系统 | Linux / WSL2 | Ubuntu 22.04+ |
| 编译 | GCC/Clang + make | GCC 11+ |

> **注意**：macOS 的 `MADV_FREE` 语义弱于 Linux 的 `MADV_DONTNEED`（延迟释放），效果可能打折。Windows 不支持，必须通过 WSL2 使用。

---

## 局限性

1. **首 token 慢**：prefill 阶段不走流式缓存，首次推理需大量页面调入
2. **不适合小模型**：模型 < 内存时直接全量加载更快
3. **上下文受限**：32GB 内存建议 `-c 4096`，开大了 KV 缓存会撑爆
4. **macOS 效果打折**：`MADV_FREE` 不立即释放页面
5. **MTP 不推荐**：实验性功能，CPU 路径收益微小

---

## 路线图

- [x] madvise 专家分页核心逻辑
- [x] 内存预算自动检测 (`/proc/meminfo`)
- [x] 路由预测预取（共现矩阵）
- [x] 使用量持久化 (`.ds4_usage`)
- [ ] 专家 top-p 剪枝 (`--topp` 参数)
- [ ] 自适应预取窗口
- [ ] 预取命中率统计

---

## 致谢

PicoMoE 基于以下两个优秀项目构建：

- **[antirez/ds4](https://github.com/antirez/ds4)** — DeepSeek V4 Flash 的高性能推理引擎。提供了完整的 CSA 压缩注意力、Metal/CUDA/ROCm 后端、分布式推理、以及所有下游工具（server、agent、bench、eval）。PicoMoE 在其 CPU 推理路径上增加了专家分页层，使大模型能运行在更小的内存上。

- **[JustVugg/colibri](https://github.com/JustVugg/colibri)** — 为 GLM 5.2 设计的显式专家分页方案，开创性地使用 madvise 按需管理 MoE 专家权重。PicoMoE 借鉴了其"内存当显存、磁盘当内存"的核心思路以及 LRU 淘汰、路由预测等关键技术。

同时也感谢 **llama.cpp / GGML** 项目和所有贡献者，没有 GGUF 格式和 mmap 机制，这一切都无法实现。

---

# PicoMoE

> Small RAM, Big Models — MoE Inference Engine with madvise Expert Paging

**PicoMoE** enables giant MoE models like DeepSeek V4 Flash (81GB) to run on machines with as little as **32GB of RAM**. The core idea: "memory as VRAM, disk as RAM" — using madvise syscalls to precisely control which expert weights stay resident in memory.

- [中文版](#picomoe)

---

## Quick Start

### Build

On Linux or WSL2:

```bash
make cpu
```

### Download Model

```bash
./download_model.sh q2-imatrix    # ~81GB
```

### Run

```bash
# Interactive chat
./ds4 -m ds4flash.gguf --cpu --ssd-streaming -c 4096 --nothink

# One-shot prompt
./ds4 -m ds4flash.gguf --cpu --ssd-streaming -p "Hello" --nothink -c 4096

# HTTP API server
./ds4-server -m ds4flash.gguf --cpu --ssd-streaming -c 4096 --host 0.0.0.0 --port 8000
```

---

## How It Works

```
┌──────────────────────────────────────────────────────┐
│  81GB GGUF file (mmap'd)                              │
│  ┌────────────┬─────────────────────────────────────┐│
│  │ Resident    │      MoE Expert Weights (bulk)      ││
│  │ ~10GB      │  E0 │ E1 │ E2 │ ... │ E255        ││
│  │            │  Paged on demand                     ││
│  └────────────┴─────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
         │                      │
         ▼                      ▼
    Always in RAM          madvise-controlled
                           ┌──────────────────────────┐
                           │ WILLNEED   async prefetch │
                           │ DONTNEED   evict cold     │
                           │ route-ahead prediction   │
                           │ usage learning & pin     │
                           └──────────────────────────┘
```

Three key mechanisms:

1. **On-demand prefetch**: After routing selects experts, `madvise(MADV_WILLNEED)` triggers async kernel readahead
2. **LRU eviction**: `madvise(MADV_DONTNEED)` frees cold expert pages untouched for 64+ decode cycles
3. **Predictive prefetch**: A co-occurrence transition matrix predicts the next layer's experts, issuing background WILLNEED while the current layer computes

---

## Features

### Model Support

| Model | Size | Supported |
|-------|------|-----------|
| DeepSeek V4 Flash | 81GB (q2) | ✅ Full |
| DeepSeek V4 Pro | Larger | ⚠️ Needs more RAM |

### Backend Support

| Backend | Status | Requirements |
|---------|--------|--------------|
| **CPU + SSD streaming** | ✅ **Primary** | 32GB+ RAM |
| Metal (macOS) | ✅ | 96GB+ unified memory |
| CUDA (NVIDIA) | ✅ | 24GB+ VRAM |
| ROCm (AMD) | ✅ | Strix Halo etc. |

### Built-in Tools

| Tool | Purpose |
|------|---------|
| `ds4` | Interactive chat / one-shot |
| `ds4-server` | OpenAI-compatible HTTP API |
| `ds4-agent` | Terminal coding agent |
| `ds4-bench` | Performance benchmarking |
| `ds4-eval` | Built-in eval harness |

---

## System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| RAM | 32GB | 64GB+ |
| Storage | SSD + 100GB free | NVMe |
| OS | Linux / WSL2 | Ubuntu 22.04+ |
| Build | GCC/Clang + make | GCC 11+ |

> **Note**: macOS `MADV_FREE` has weaker semantics than Linux `MADV_DONTNEED` (lazy reclamation), so results may be suboptimal. Windows is not supported natively — use WSL2.

---

## Limitations

1. **Slow first token**: The prefill path does not use streaming cache; initial inference requires significant page-in
2. **Not for small models**: Models that fit entirely in RAM are faster with full preload
3. **Limited context on 32GB**: Recommended `-c 4096`; larger contexts exhaust RAM via KV cache
4. **macOS penalty**: `MADV_FREE` does not immediately reclaim pages
5. **MTP not recommended**: Experimental speculative decoding yields minimal gains on CPU

---

## Acknowledgments

PicoMoE builds on two outstanding projects:

- **[antirez/ds4](https://github.com/antirez/ds4)** — A high-performance inference engine for DeepSeek V4 Flash, providing the complete CSA compressed attention implementation, Metal/CUDA/ROCm backends, distributed inference, and all downstream tools. PicoMoE adds an expert paging layer on top of its CPU inference path.

- **[JustVugg/colibri](https://github.com/JustVugg/colibri)** — An explicit expert paging scheme for GLM 5.2 that pioneered the use of madvise for on-demand MoE expert management. PicoMoE adapts its core ideas (LRU eviction, route prediction) for DeepSeek V4 Flash.

Also thanks to **llama.cpp / GGML** and all contributors — without the GGUF format and mmap mechanism, none of this would be possible.
