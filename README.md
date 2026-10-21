# coolmem

轻量记忆服务：FastAPI + SQLite，提供消息写入与按用户隔离的全文检索；单进程运行，数据存本地文件，不依赖外部数据库或缓存；embedding 服务为必需依赖。

中文检索由 FTS5 + cutword 分词 + CJK 单字/双字扩展完成；写入与检索都要求 embedding 服务可用。

## 环境要求

- Python 3.12+
- 无需外部数据库或缓存服务，数据保存在本地 SQLite 文件
- 必需：OpenAI 兼容的 embedding HTTP 服务（如本机 llama.cpp）；未配置或禁用时服务拒绝启动，运行期不可用时 `/add`、`/search` 返回 503

## 安装

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

## 配置

配置通过环境变量提供；token 必填，可来自 `COOLMEM_BEARER_TOKEN` 或当前目录 `env` / `.env`。embedding 配置同样必填，见下文：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `COOLMEM_HOST` | `0.0.0.0` | 监听地址 |
| `COOLMEM_PORT` | `8005` | 监听端口 |
| `COOLMEM_BEARER_TOKEN` | 无（token 必填） | `/add`、`/search` 的 Bearer token，至少 16 字符、不含空白；不设时读 `env` / `.env` |

token 未提供、少于 16 字符或含空白字符时服务拒绝启动；环境变量优先于 `env` / `.env`；token 不要提交到 git。

### 本地 token 文件（可选）

不想用环境变量时，在当前目录放 `env`（或 `.env`），只写 token 原文，不要写 `COOLMEM_BEARER_TOKEN=`，不要加引号或多行。启动时如果没有设置 `COOLMEM_BEARER_TOKEN`，会先读 `env`，再读 `.env`。两个文件都已加入 `.gitignore`；建议 `chmod 600 env`。

生成一个随机 token（不回显）：

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > env
chmod 600 env
```

也可以直接写你自己的固定字符串：

```bash
printf '%s\n' '你的固定字符串' > env
chmod 600 env
```

数据目录固定为工作目录下的 `./data`（SQLite 文件 `data/coolmem.sqlite3`），请求体大小上限 4,000,000 字节，两者都不通过环境变量修改。

### Embedding（必需）

`config/config.example.json` 是模板；复制为 `config/config.local.json`（已在 `.gitignore` 中）后按本机模型填写。代码只读取工作目录下的 `config/config.local.json`；文件缺失、`enabled=false` 或配置不合法时服务拒绝启动。项目只调用这个 HTTP 服务，不启动/管理模型进程；服务与模板由各机器自行准备。

- `url`：OpenAI 兼容的 `/v1/embeddings` 完整地址（llama.cpp、vLLM、LM Studio 等都可用）。
- `model` / `dimension`：本机模型名称与向量维度。
- `query_template` / `document_template`：本机模型的输入模板，必须分别包含 `{query}` / `{document}`；不同模型的 instruction 不同，所以只放本地文件。
- `batch_size` / `batch_max_chars`：每批条数上限与总字符预算；长文本会自动拆成多批。
- `max_input_chars`：单条文本送入 embedding 前的截断长度；原文仍完整存储。
- `timeout_seconds` / `retries`：HTTP 超时与重试次数。
- `api_key_env`：可选，填环境变量名，不要填 key 本身；需要鉴权的服务在环境里设置该变量。
- `tiebreak_weight`：dense 分数参与 RRF 同分裁决的权重，默认 0。

向量按 `(message_id, space)` 存进 SQLite；`space` 由 model、dimension、query/document 模板哈希得到。换模型或模板后旧向量不会被误用；旧数据需要重新写入/回填后才能被检索到。

Embedding 不可用时：`/add` 返回 503 `service_unavailable`，不写入任何消息或向量；`/search` 返回 503；不会退化为纯词法。错误响应不包含 query、message、URL 或模板。

### Rerank（可选，默认关闭）

`config/config.local.json` 的 `rerank` 段接一个 cross-encoder 精排服务（llama.cpp `--reranking` 的 `/v1/rerank`、OpenRouter 的 `/rerank` 都兼容）：

- `enabled`：`true` 时对检索候选做精排；配置缺失或 `false` 时保持纯 RRF 排序。
- `url` / `model`：完整 `/v1/rerank` 地址与模型名，只放本地文件。
- `timeout_seconds` / `retries` / `batch_size`：HTTP 超时、重试与单次请求的文档条数。
- `failure_threshold` / `cooldown_seconds`：连续失败几次后熔断、熔断多久（默认 3 次 / 60 秒）；熔断期间 `/search` 直接走纯 RRF，不再等待精排。
- `api_key_env`：可选，填环境变量名。

返回条目的长度截断由 `Policy.max_return_chars` 控制（默认 0 = 不截断）：开启后超长条目保留头部 3/4 与尾部 1/4，中间以 `...` 省略。短条目不受影响；它只改返回正文，不改索引与排序。

精排失败（服务不可达、返回结构不符）时**自动降级为原排序**，`/search` 不会因此失败。精排深度、混合权重与文档截断由 `Policy` 的 `rerank_depth` / `rerank_blend` / `rerank_doc_chars` 控制（默认 200 / 1.0 / 2400）；`rerank_blend=1.0` 表示完全按精排分数排序。

返回正文的邻居扩展由 `Policy.neighbor_message_chars`（默认 0，即关闭）控制：开启时命中条目会附带同 session 前一条的尾部与后一条的头部各 N 字符，邻居超过 `neighbor_skip_chars`（默认 640）时跳过。它只改返回正文，不改索引与排序。实测它会提高 reached@K（gold 落进邻居文本）但降低端到端答题准确率，因此默认关闭。

## 启动

```bash
.venv/bin/python -m app.main
```

默认监听 `0.0.0.0:8005`（所有网卡）；本机自测可设 `COOLMEM_HOST=127.0.0.1`。启动前必须提供 token：设置 `COOLMEM_BEARER_TOKEN`，或在当前目录放 `env` / `.env`（见上）。未提供、过短或含空白字符时服务拒绝启动；公网部署建议 bind `127.0.0.1` 并由反向代理转发。启动前必须具备有效的 embedding 配置；缺失或 `enabled=false` 会拒绝启动。首次启动自动建表并迁移到当前 schema 版本，无需手工操作。前台运行时按 `Ctrl+C` 停止。

## 接口

### 鉴权

`/add`、`/search` 需要 `Authorization: Bearer <token>`，token 由启动时的 `COOLMEM_BEARER_TOKEN` 或当前目录 `env` / `.env` 决定；`/health` 不需要。缺失或错误返回 401 `unauthorized`。

### POST /add

```json
{
  "request_id": "req-1",
  "user_id": "u1",
  "session_id": "s1",
  "messages": [
    {"role": "user", "content": "我叫张三", "timestamp": 1750000000000}
  ]
}
```

- `role` 只能是 `user` 或 `assistant`；`timestamp` 可选，非负 Unix 毫秒，缺省用服务端当前时间。
- 带 `timestamp` 的消息，`content` 会写成 `<数字时间> <中文时间> <英文时间> <原文>`；原文另存 `source_content` 用于幂等比对。时间按 UTC 生成。例：`2022-01-21 19:31 2022年1月21日 星期五 January 2022 Friday 07:31 PM 原文`。
- 幂等键是 `(user_id, request_id)`：同键同 payload 重放返回 200 且不重复写入；同键不同 payload 返回 400 `request_id_payload_conflict`，响应 `details` 里给 `mismatch`（`message_count` / `session_id` / `position@N` / `role@N` / `content@N` / `timestamp@N`）与 `existing_messages`、`incoming_messages`。
- 冲突报文格式：`payload conflict on request_id='<id>' user_id='<uid>': mismatch=<field> existing_messages=<n> incoming_messages=<m>`。
- 上限：每次 1~500 条消息；`request_id` / `user_id` / `session_id` ≤ 256 字符；`content` ≤ 100,000 字符且不能全空白。

### POST /search

```json
{"query": "助手叫什么名字", "user_id": "u1", "top_k": 5}
```

- `query` 非空且 ≤ 300,000 字符；`top_k` 必须为正，超过 100 按 100 处理。
- 超长 query 不会直接 422：系统会生成一份检索副本（问题/指令锚点 + 首尾上下文），原文不落库，返回内容不受影响。
- 只检索同一 `user_id` 的消息，其他用户不可见。
- 响应：`{"data": [{"id": "mem_12", "content": "...", "score": 0.83, "created_at": "..."}]}`

### GET /health

正常返回 `{"status": "ok"}`；数据库不可读时返回 503 `service_unavailable`。健康检查只读数据库，不探测 embedding 服务；embedding 不可用时 `/add`、`/search` 返回 503。

### 错误格式

```json
{"error": {"code": "invalid_input", "message": "...", "trace_id": "...", "details": {}}}
```

| 情形 | 状态码 | code |
|---|---|---|
| 缺失或错误的 Bearer token | 401 | `unauthorized` |
| 请求体超过 1,000,000 字节 | 413 | `payload_too_large` |
| schema 校验失败 | 422 | `validation_error` |
| 业务规则拒绝 | 422 | `invalid_input` |
| embedding 服务不可用 | 503 | `service_unavailable` |
| 未知异常 | 500 | `internal_error` |

响应头带 `x-request-id`。

## 检索实现

0. **长 query 裁剪**：query 超过 `Policy.search_query_chars`（默认 12,000）时，先抽问题/指令行放到检索副本前部，再取首 40% / 尾 60% 上下文；切口对齐换行或句子边界。原文不存储、返回内容不变。精排未在默认链路；接入时 query/document 预算单独算，不复用 embedding 的 12k。
1. **索引侧文本**（写入时生成）：原文 + 日期/数字的等价写法展开 + 每个 CJK 连续段的单字、双字与分词结果；同时记录同一次 add 内前后各 3 条消息的正文作为邻居文本。
2. **词法检索**：FTS5 上的 BM25（`search_text` 权重 1.0、`neighbor_text` 0.5、`role` 0.2）；先 AND 再 OR 补齐名额。
3. **子串检索**：trigram 索引覆盖分词切不中的片段；候选深度只有 `max(1, int(top_k × 0.5))`（`Policy.substring_depth_ratio`），而词法/向量通道为 `max(top_k, 200)`。
4. **向量检索（必需）**：Add 时先调 embedding 服务，成功后才把消息和 float32 向量在同一事务里写入（`message_vectors`）；Search 时对 query 调同一服务，按 `user_id` + `space` 做余弦相似度取候选。
5. **查询门控（可选，默认关闭）**：开启 `Policy(lane_gating=True)` 后，按 query 形态选 `dense` / `hybrid` / `literal` 三条车道。`dense` 只跑向量路；`hybrid` 三路都跑并放大词法权重；`literal` 只跑词法与子串。门控只看 query 本身（引号、代码/ID、日期/数字、短实体问句、词数、问句结构），阈值在 `app/gating.py` 的 `GateConfig`。默认质量优先走等权三路。
6. **融合**：RRF + 车道权重 + BM25 归一化同分裁决 + 查询词覆盖率加成。开启门控时 hybrid 权重为 dense 1 / lexical 8 / substring 8；literal 为 lexical 1 / substring 1。
7. **不回退**：embedding 不可用或调用失败 → 503，不转词法；dense 返回空就是空结果。仅 `literal` 车道在词法无命中时回退 dense（仍要求 embedding）；候选深度不变。
8. **指令边界分解**：query 含“核心问题 + 格式指令”两段时（无选项、空行分隔、次段含 `Use only / Return all / Document title` 等标记），对核心段再检索一次，按 RRF 与完整 query 合并。只增加候选，不降深度；`Policy.instruction_decomposition` 可关。
9. **证据包窗口**：最终排序后，把每条命中扩展为同 session 前后各 `context_window_turns` 条消息拼接的正文；种子 id/score 不变，不加新索引、零 LLM。默认 2，设 0 关闭。
10. 其余阈值集中在 `app/policy.py`。

**期望构成（诊断用，不作为验收）**：开启门控时，`top_k=100` 的最终结果预期约为仅语义 75%、语义+词法 15%、仅词法 10%。默认质量优先配置是 `lane_gating=False`，不保证该构成；构成只用来发现“哪类 query 走错车道”。验收看 K=100 的 reached@1/@3/@10/@100 和端到端答对率；MRR 只进诊断表。

## 数据与备份

- 数据文件默认 `./data/coolmem.sqlite3`，首次启动自动创建，运行在 WAL 模式。
- **不要在服务运行时直接复制主文件**，用 SQLite 在线备份接口：

```bash
.venv/bin/python - <<'PY'
import sqlite3, datetime, pathlib
src = sqlite3.connect("data/coolmem.sqlite3")
dst_path = pathlib.Path("data") / f"backup-{datetime.date.today():%Y%m%d}.sqlite3"
dst = sqlite3.connect(dst_path)
with dst:
    src.backup(dst)
dst.close(); src.close()
print("backup ->", dst_path)
PY
```

- 恢复：停服 → 用备份文件替换 `data/coolmem.sqlite3`（同时删除 `-wal` / `-shm`）→ 启动。
- 数据库迁移只增不改，`_SCHEMA_VERSION` 单调递增；版本不符时启动会直接报错退出。

## 日志

日志每行格式为 `时间 方法 路径 状态码`，不记录 payload、trace_id、客户端地址与耗时；日志不做自动轮转，长期运行需自行配合 logrotate 或定期截断。

## 安全说明

- **固定 Bearer token 鉴权。** `/add`、`/search` 需要 `Authorization: Bearer <token>`；token 来自 `COOLMEM_BEARER_TOKEN` 或当前目录 `env` / `.env`（环境变量优先），不得写进代码、文档、日志或 git。`env` / `.env` 已在 `.gitignore` 中。`/health` 免鉴权。默认监听 `0.0.0.0:8005`；公网部署建议 bind `127.0.0.1` 并由反向代理/防火墙限制来源。
- **模型配置只放 `config/config.local.json`。** 端点、模型名、维度、query/document 模板和可选 key 的环境变量名都不要写进代码、文档、日志或提交；文件已在 `.gitignore` 中。
- 只暴露 `/add`、`/search`、`/health` 三个路由，字段名、响应结构与错误码是对外契约。

## 开发约定

代码零注释（`#` 注释与 docstring 均禁止），`python3.14 script/check_no_comments.py` 会校验注释、docstring 与 markdown 白名单。硬性约束与实现细节见 `AGENT.md`。
