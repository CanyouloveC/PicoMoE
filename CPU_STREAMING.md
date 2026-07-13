# CPU Expert Streaming — 小内存运行大模型

## 背景

DeepSeek V4 Flash 的 q2 量化模型约 **81GB**，官方要求 ≥96GB 内存。如果你只有 **32GB 内存**（比如笔记本），传统方案无法加载。

本项目借鉴 [colibri](https://github.com/JustVugg/colibri) 的显式专家分页思路，在 ds4 的 CPU 推理路径上实现了一套 **madvise 驱动的专家页面流式缓存**，让 81GB 的模型能在 32GB 内存上运行。

## 核心原理

```
┌──────────────────────────────────────────────────┐
│  81GB GGUF 文件 (mmap)                            │
│  ┌─────────┬────────────────────────────────────┐ │
│  │ 常驻部分 │        MoE 专家权重 (主体)         │ │
│  │ ~10GB   │  Expert 0 │ Expert 1 │ ... │ E255 │ │
│  └─────────┴────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
   始终在内存           madvise 按需管理
                       ┌─────────────────┐
                       │ WILLNEED  预取   │ ← 路由后立即通知内核
                       │ DONTNEED  淘汰   │ ← LRU 冷专家释放
                       │ route-ahead 预测 │ ← 提前预取下一层
                       └─────────────────┘
```

- **MADV_WILLNEED**：推理时，路由选定专家后立即通知内核异步预读对应页面
- **MADV_DONTNEED**：对长期未使用的冷专家页面释放内存
- **Route-ahead 预取**：利用专家共现矩阵，在处理当前层时预测并预取下一层的专家
- **使用量学习**：跨会话记录每个专家的使用频率，热启动时预加载热门专家

## 与 colibri 的对比

| 特性 | colibri (GLM 5.2) | 本方案 (DS4 Flash) |
|------|--------------------|---------------------|
| 分页粒度 | 专家级 | 专家级 |
| 内存管理 | madvise | madvise |
| 预取策略 | 路由预测 (PILOT) | 共现矩阵 + route-ahead |
| 使用量持久化 | 无 | ✅ `.ds4_usage` 二进制文件 |
| 内存预算 | 手动配置 | 自动从 `/proc/meminfo` 检测 |
| 侵入性 | 需修改推理代码 | 零侵入 matmul kernel |

## 文件结构

```
新增文件:
  ds4_cpu_stream.h        (~135行)  公开 API
  ds4_cpu_stream.c        (~450行)  核心实现

修改文件:
  ds4.c                   (+272行)  桥接层 + 解码路径集成
  Makefile                (+13行)   编译目标
  ds4_cli.c               (+3行)    --cpu-usage-file 参数
  ds4_help.c              (+5行)    帮助文本

文档:
  CPU_STREAMING.md         本文件
```

## 使用方法

### 1. 编译

在 WSL2 或 Linux 上：

```bash
make cpu
```

生成 `ds4`、`ds4-server`、`ds4-bench`、`ds4-eval`、`ds4-agent` 五个二进制文件。

### 2. 下载模型

```bash
./download_model.sh q2-imatrix    # 下载 81GB q2 量化模型
```

### 3. 运行

```bash
# 交互对话
./ds4 -m ds4flash.gguf --cpu --ssd-streaming -c 4096 --nothink

# 单次提问
./ds4 -m ds4flash.gguf --cpu --ssd-streaming -p "你好" --nothink -c 4096

# HTTP API 服务器
./ds4-server -m ds4flash.gguf --cpu --ssd-streaming -c 4096 --host 0.0.0.0 --port 8000
```

### 4. 关键参数

| 参数 | 说明 |
|------|------|
| `--cpu` | 强制 CPU 后端 |
| `--ssd-streaming` | 启用 SSD 流式加载（必须） |
| `-c, --ctx N` | 上下文窗口，32GB 内存建议 4096 |
| `--nothink` | 关闭思考模式，加速响应 |
| `--cpu-usage-file PATH` | 使用量缓存文件，默认 `<模型名>.ds4_usage` |

## 技术细节

### 内存预算自动检测

```
MemAvailable (来自 /proc/meminfo)
  → 预留 2GB 系统缓冲
  → 取剩余 80% 作为专家缓存预算
```

### 淘汰策略 (LRU)

- 每个专家维护 `access_count` 和 `last_clock` 两个字段
- 时钟每次解码 token 递增
- 超过 64 个时钟周期未访问的专家被标记为冷专家
- 每次最多淘汰 `max_cold` 个专家，避免抖动

### 路由预测

```
Layer L expert selections
        │
        ▼
  共现矩阵 T[expert_A * n_experts + expert_B]
        │
        ▼
  Layer L+1 预测的 top-k 专家
        │
        ▼
  MADV_WILLNEED (后台预取)
```

在处理 layer L 的 FFN 时，根据历史共现概率预测 layer L+1 可能选择的专家，提前发起预读。当 layer L+1 实际需要时，页面大概率已在内存中。

### 使用量持久化

运行过程中，每个专家的访问次数被实时记录。退出时写入 `.ds4_usage` 文件（二进制格式，带魔数校验）。下次启动时自动加载，热门专家在模型加载阶段就被预取到内存中。

## 系统要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| 内存 | 32GB | 64GB+ |
| 磁盘 | 81GB 模型 + 可用空间 | SSD（NVMe 更佳） |
| 系统 | Linux / WSL2 | Ubuntu 22.04+ |
| 编译 | GCC/Clang + make | GCC 11+ |

注意：macOS 上的 `MADV_FREE` 比 Linux 的 `MADV_DONTNEED` 语义更弱（延迟释放），效果可能不如 Linux。

## 局限性

1. **首 token 延迟**：prefill 阶段不走流式缓存，第一次推理需要较多页面调入
2. **不适合小模型**：模型本身小于内存时直接全量加载更快
3. **macOS 效果打折**：`MADV_FREE` 不会立即释放页面
4. **Windows 不支持**：必须通过 WSL2 使用

## 路线图

- [x] madvise 专家分页核心逻辑
- [x] 内存预算自动检测
- [x] 路由预测预取
- [x] 使用量持久化
- [ ] 专家 top-p 剪枝 (`--topp` 参数)
- [ ] 自适应预取窗口大小
- [ ] 预取命中率统计与上报

## 致谢

- [antirez/ds4](https://github.com/antirez/ds4) — DeepSeek V4 Flash 推理引擎
- [JustVugg/colibri](https://github.com/JustVugg/colibri) — 提供了专家分页的核心思路
