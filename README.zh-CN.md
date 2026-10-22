# Mumble Rhyme

[English](README.md) · 中文

**你决定哪些声音可以押；工具帮你找到说得通的英文。**

给非母语英语创作者用的本地歌词助手：用中文说出意思，获取英文补词或整句建议；由你决定哪些声音可以通押。提供 Python API、JSON CLI 和 stdio MCP，供人类或旗舰模型调用。

**状态：0.3.1 可运行版本。** 发音规则、排序、MCP 已验证；本机 RTX 5080 的真实 Qwen3.5-9B 与 4B NF4 生成及评分已通过。实测记录见 [验证记录](docs/verification.md)。

## 模型与范围

支持显式选择 **Qwen3.5-9B 或 Qwen3.5-4B**。默认仍为 9B（`c202236235762e1c871ad0ccb60c8ee5ba337b9a`），`--model 4b` 使用官方 4B（`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`）。每个模型的权重与 tokenizer 绑定到各自同一提交。不会偷偷换成较小模型。默认在 CUDA 上用 bitsandbytes NF4 4-bit 加载；9B 原始下载约 19.31 GB，4B 约 9.32 GB，量化在本地加载时完成。量化不改变词表，但会影响数值精度。无需音频输入。

Windows 首次加载会将已下载权重流式重排成较小分片，以绕过 safetensors 大文件映射导致的分页内存峰值；不改变张量数值。原始缓存保留，因此额外占用与所选原始权重大小接近的磁盘空间，之后会复用重分片缓存。无需修改系统分页文件。

首版发音评分针对**英文**；创作意图可以用中文。中文韵母评分尚未实现。可用来探索旋律说唱、近韵、情绪化口语表达；不包含艺术家的歌词训练集或风格微调。

## 安装（Windows / RTX 5080）

在此项目目录运行：

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -e '.[model,dev]'
.venv/Scripts/python.exe -m mumble_rhyme download
```

核心功能可单独安装 `uv pip install --python .venv/Scripts/python.exe -e .`，不需要 PyTorch。Python 3.10+。其他平台应先按自己的硬件安装兼容的 PyTorch；`--quantization none` 可禁用量化，需要更多内存。当前默认 NF4 路径要求 CUDA。依赖快照见 `requirements-lock.txt`（本机 Windows / Python 3.10 环境）；模型依赖是可选项。

显式使用 4B：

```powershell
.venv/Scripts/python.exe -m mumble_rhyme --model 4b download
.venv/Scripts/python.exe -m mumble_rhyme --model 4b --profile examples/mumble.yaml suggest --request examples/suggest.json
.venv/Scripts/python.exe -m mumble_rhyme --model 4b --profile examples/mumble.yaml serve
```

4B MCP 配置见 [Windows](examples/mcp.windows.4b.json) / [POSIX](examples/mcp.posix.4b.json)。每个服务进程固定使用启动时选定的模型，工具调用不会自动切换。

已安装后的使用：

```powershell
# 不启动模型：查看自定义 i/e 规则对 sit/set 的影响
.venv/Scripts/python.exe -m mumble_rhyme --profile examples/mumble.yaml compare sit set

# 不启动模型：从词典找候选。整个词典都软评分，返回排名靠前的一部分。
.venv/Scripts/python.exe -m mumble_rhyme find night --limit 12

# 启动 Qwen：根据上下文比较你或旗舰模型提出的几个表达
.venv/Scripts/python.exe -m mumble_rhyme --profile examples/mumble.yaml rank --request examples/rank.json

# 启动 Qwen：从中文意思生成英文整句，再按规则排序
.venv/Scripts/python.exe -m mumble_rhyme suggest --request examples/suggest.json

# 查看固定模型配置，不加载模型
.venv/Scripts/python.exe -m mumble_rhyme status
```

`rank` / `suggest` 也接受 `--request -`，从标准输入读取 JSON。所有正常输出都是 JSON；日志和错误写到 stderr。MCP 会常驻并复用已加载模型；每次 CLI 调用是新进程，需要重新加载。

4B 实测（RTX 5080 / torch 2.8.0+cu128）：缓存已有时，真实生成及评分脚本约 12.56 秒、正式 MCP 调用约 14.68 秒；PyTorch 峰值已分配显存约 3.217 GiB。这是整轮功能测试，非每次生成延迟或最低显存承诺。原始结果与 9B 内存问题的修复记录见 [验证记录](docs/verification.md)。

## 你的“通押自定义”

这份配置里的 `i`、`e` 是你命名的符号，不默认等于字母名称或某个 IPA 音。用示例词把符号钉到读音上：

```yaml
name: my-mumble
aliases:
  i: {vowel_of: sit}   # IH
  e: {vowel_of: bed}   # EH
  ee: IY
vowel_rules:
  - sounds: [i, e]
    weight: 0.95
    symmetric: true
  - sounds: [i, e]
    position: 2       # 只影响倒数第二个音节；1 = 最后一个音节
    weight: 0.8
syllables: 2          # 比较末尾两个音节
unknown_weight: 0.7
pronunciations:
  luv: [L AH1 V]      # 你实际准备怎么唱；多个读音可写多个字符串
```

- 权重范围 `(0, 1]`：1 是你认为完全可接受，0.95 是轻微降权。未配置的音对继续使用默认相似度，不会被判死刑。默认相似度是手工启发式，并非训练出的听感概率。
- 规则只作用于指定音对；`i ↔ e`、`e ↔ a` 不会推出 `i ↔ a`。`symmetric: false` 时顺序为 `[候选音, 目标音]`。
- 位置规则优先于通用规则；同等具体的重复规则采用最后一条。每个请求可以传 `profile` 临时覆盖配置；`aliases` / `pronunciations` 按键合并，`vowel_rules` 整组替换。不修改已保存文件。
- `vowel_importance` 控制元音在音节评分中的比重，默认 0.75；剩余部分是尾部辅音的编辑距离权重。`stress_mismatch_weight` 默认 0.93，弱化重音不同的匹配。
- 别名可直接使用 ARPAbet 元音：`AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW`。例词默认取词典第一个读音的最后元音，可用 `syllable_from_end` 指定倒数位置。
- `pronunciations` 支持空格分隔的 ARPAbet，元音可带 `0/1/2` 表示重音。可为俚语、名字或你特定的唱法覆盖读音。初始发音库是美式英语 CMUdict。

## 怎么与旗舰模型配合

把 [examples/mcp.windows.json](examples/mcp.windows.json) 中的 server 配置合并进支持 stdio MCP 的客户端。示例使用占位绝对路径；请替换成自己的 Python 和项目路径。不同客户端配置入口可能不同。

服务启动等价于：

```powershell
.venv/Scripts/python.exe -m mumble_rhyme --profile examples/mumble.yaml serve
```

提供六个工具：

| 工具 | 输入与用途 | 本地模型 |
| --- | --- | --- |
| `compare_rhyme` | 比较词或短语，解释每个末尾音节的元音、尾辅音和规则 | 不需要 |
| `find_rhymes` | 词频与音韵软排序，返回同音拼写备选；保留纯音韵模式 | 不需要 |
| `search_lyrics` | 默认 20×20 token 树搜索，返回真实组合概率与软押韵加权 | 需要 |
| `rank_candidates` | 给候选字符串、左右文、韵脚，结合语境与发音排序 | 默认需要；可 `use_model=false` |
| `suggest_lyrics` | 给中文意思及上下文，生成英文补词或完整一行并排序 | 需要 |
| `model_status` | 查看固定提交、量化、加载状态 | 不需要 |

可对旗舰模型这样说：

> 我想表达“深夜还想给前任打电话，但又不想承认”。帮我写一行自然英文歌词，末尾参考 night。我允许 IH/EH 轻微通押。先用 suggest_lyrics 找方向；再根据我的意思提出几个不同写法，用 rank_candidates 比较。解释词义、口语用法，以及哪里需要我唱的时候稍微弯一下音。

`left` / `right` 是空白处两侧的已有内容，`candidates` 只放要填入的片段。需要改写整句就把左右文留空，候选填完整一行。`target_syllables` 是**填入片段**的音节数软偏好；不满足也保留。中间填空时，发音比的是填入片段的末尾，而不是 `right` 的末尾。

`strength` 默认 2.0，越大越重视韵脚；0 关闭韵脚加权（独立的音节数偏好仍会生效）。刚开始可试 0.5–3。没有任何韵脚硬门槛。

## 分数到底代表什么

对已经给出的有限候选集，工具计算：

```text
score = mean_token_log_likelihood + strength * log(rhyme_weight) + log(syllable_weight)
rank_weight = softmax(score over the supplied candidate set)
```

- 词可能拆成多个 token。评分会把完整文本重新 tokenize，再按字符偏移定位要评分的候选和右文；不会把某个 token ID 当成一个发音单词。跨过左文边界的 token 也会算入。
- `lm_sum_logprob` 是实际求和的 token 对数似然；`lm_mean_logprob` 用于排序以减轻短文本偏好。右文也参与评分，因此能比较中间填空的衔接。均值与右文长度会影响不同长度候选的排序。
- `rank_weight` 只是此候选集中的相对排序权重，**不是**一句歌词好听/自然/合意的概率。增加或移除候选后它会变化。没有训练过主观音乐听感模型。
- 无模型模式明确标记为 `phonetic_only`，不会假装评估了语义。CLI / Python 可传 `model_scores` 使用外部给出的有限分数；MCP 不开放这个内部评测入口。
- 多个读音取最佳发音匹配，并返回选中的音素。每个短语最多考虑 32 个发音组合，超出会返回 `pronunciation_search_truncated=true`。未知尾词得到 `unknown_weight`，保留且标记；未知词后面有已知尾词时可继续比较已知结尾。
- 所有**提交的**候选都保留。生成本身只有有限预算，`find_rhymes` 也只返回 top N；这不意味着搜索了所有可能歌词。可用自己的候选调用 `rank_candidates` 扩大搜索。
- 首版按元音分段比较末尾 N 个音节，尾辅音归属是启发式，不是专业音节切分。没有自动推断连读、方言、拖音或实际演唱发音；请用覆盖读音表达它们。

## 开发验证

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m build
.venv/Scripts/python.exe scripts/smoke_model.py
```

模型 smoke 脚本均支持 `--model 4b` / `--model 9b`；4B 结果写入独立的 `docs/model-smoke-4b.json` 与 `docs/mcp-model-smoke-4b.json`，9B 原文件名保留。测试会执行所选真实模型的生成和似然评分。常规测试不下载模型、不要求 GPU；包含真实 stdio MCP 子进程交互。模型缓存沿用 Hugging Face 默认位置，可通过 `HF_HOME` 由用户设置。模型和词典各自遵循其上游许可；CMUdict 许可随 Python 依赖分发。

## 开源与贡献

代码采用 [GPL-3.0-or-later](LICENSE)，与当前 cmudict Python 封装一致。模型和词典数据保持各自上游许可，见 [第三方声明](THIRD_PARTY_NOTICES.md)。仓库不附带模型权重。

[安装指南](docs/installation.md) · [音对规则](docs/rhyme-rules.md) · [MCP/API](docs/api.md) · [写作场景对比](docs/workflows.md) · [贡献指南](CONTRIBUTING.md)

当前是候选生成与软排序：生成器参考目标韵脚，用户音对规则在排序阶段生效。可自行提供候选以探索生成器未覆盖的非常规押法。

## 0.2.1 内存修复

已修复 Windows 大张量读取产生额外副本的峰值、失败异常链保留部分模型，以及 cuBLAS 工作区阻止大块显存归还的问题。9B、4B 均通过模拟失败后恢复，并在同一进程内连续加载/评分/卸载两轮；使用显式完整缓存释放后，实测为 1 KiB 已分配、2 MiB 已预留。Python 可调用 `service.model.unload()` 释放模型；若需清理进程级 cuBLAS 缓存，可在同进程其他 CUDA 工作已停止时使用 `release_runtime_cache=True`；MCP 正常在会话内常驻模型，停止服务进程会释放其运行时上下文。详见 [内存审计](docs/model-memory.md)。

## 0.2.2 首次生成与错误恢复

首次 NF4 加载会先初始化 cuBLAS/BF16 运算路径。生成或评分中遇到显存分配错误，会释放模型和异常堆栈中的临时张量，避免报错后继续占用整份模型。不会自动换模型或假装返回成功。`model_status` 新增 `package_version` 和 `last_resource_error`。

更新后请在 Cherry Studio 中停止/重新启动此 MCP，再确认 `package_version` 为 `0.3.0`；运行中的旧进程不会自动更新代码。两种模型已验证以用户的城市灯光请求冷启动直接生成，并验证模拟分配错误后的同进程重试。原始驱动错误在重启后未稳定复现，不能把模拟错误当成原生故障复现，详见 [调试记录](docs/suggest-reliability.md)。

## 0.3.0：真实概率树搜索与实用找词

新增第六个 MCP 工具 `search_lyrics`，对应 CLI `search`。默认每轮保留 20 条路径，每条扩展前 20 个 token，累加真实全词表条件对数概率；完整词边界用自定义软通押引导搜索，最终返回路径概率、押韵权重、加权结果和截断标记。没有语义/自然度主观打分。

```powershell
.venv/Scripts/python.exe -m mumble_rhyme --model 4b --profile examples/mumble.yaml search --request examples/search.json
```

`beam_width`、`branch_top_k` 默认均为 20；显存批量 `microbatch_size` 默认 1，不缩减搜索宽度。`length_normalization=0` 保留原始组合概率的排序，天然偏向短表达；可显式调整最短 token 长度或归一化。到最大长度时可能截在词内，会标明未确认词边界。搜索是有限预算，不能保证找到全局最优。该接口首版仅向前续写，无右文；中间填空仍可用 rank_candidates。

`find_rhymes` 默认 balanced：综合词频与软音韵，折叠同音拼写并返回 alternatives；`mode=phonetic` 保留纯音韵模式。词频不改变你的通押分数，不禁用生僻词；常见人名和网络词仍可能出现。

[算法、API 与实测](docs/probability-search.md)。更新后重启 MCP，通过 model_status 确认运行版本为 0.3.0；工具列表应有六项。

## 0.3.1：Windows 大块预分配修复

已精确定位 9B 的 7.22 GiB 单次申请来自 Transformers 的可选 allocator warmup，不是真实权重或本次 attention fallback。现在仅将这一步预热限制为 256 MiB，不改模型、真实张量形状或 20×20 搜索宽度，不修改全局依赖函数。

另发现系统 GraphicsPerfSvc 宿主占用约 54.4 GiB 提交内存，经用户授权重启并退出旧宿主后，原生 7.22 GiB 分配也成功了。因此不能把该现象一概归为固定 4 GiB 驱动上限。项目修复避免不必要的大申请，系统本身仍需足够提交余量。

9B、4B 的默认搜索均在强制拒绝大于 4 GiB 的工厂申请的测试条件下通过；实际 fallback 单层最大搜索形状（批量8/长度2048）的 pairwise masked_fill 是 128 MiB。详见 [完整调查与实测](docs/windows-allocation.md)。更新后重启 MCP，确认 package_version=0.3.1；model_status.loading_warmup 会显示预热原始大小和实际大小。
