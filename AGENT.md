# AGENT.md

面向接手本仓库的 agent 与开发者。**先读第 1 节，它是硬性约束，不可协商。**

## 1. 硬性约束

1. **任何代码都不许写注释。**
   - Python：`#` 注释、模块/类/函数 docstring 一律禁止。删掉 docstring 后若函数体为空，写 `pass`。
   - 配置文件与脚本：`.gitignore`、`.toml`、`.sh`、`.cfg`、`.ini` 里的 `#` 注释行同样禁止。
   - 需要解释的行为写在 `AGENT.md`、`README.md`、`history.md` 里，不要写回代码。
   - **每次改动后必须运行 `python3.14 script/check_no_comments.py`**，只有输出 `OK: no comments or docstrings` 才算通过。
   - 不要用"临时调试注释、稍后删"绕过；也不要靠 `# noqa`、`# type: ignore` 压制告警，改代码本身。
2. **接口契约不可破坏。** 只暴露 `POST /add`、`POST /search`、`GET /health` 三个路由，和 https://agentmemories.ai/api-guide 一致；字段名、响应结构、错误码是对外契约，改动前先确认兼容。
3. **接口鉴权**：`/add`、`/search` 只接受 `Authorization: Bearer <token>`，token 从 `COOLMEM_BEARER_TOKEN` 或当前目录 `env` / `.env` 读取（环境变量优先），至少 16 字符且不含空白；未提供或不合格时服务拒绝启动。token 不得出现在代码、文档、日志、错误响应或 git 中；`/health` 免鉴权。默认监听 `0.0.0.0:8005`，公网部署应 bind `127.0.0.1` 并由反向代理/防火墙限制来源。
4. **仓库只允许三个 markdown 文件：`AGENT.md`、`README.md`、`history.md`。** 不得新增其他 `.md`（包括子目录里的 README）
5. **模型配置只放 `config/config.local.json`**（已 gitignore）。端点、模型名、维度、query/document 模板都属于本机模板，不要写进代码、README、日志或提交；缺失、`enabled=false` 或配置不合法时服务拒绝启动；运行期 embedding 不可用时 `/add`、`/search` 返回 503，不退化词法。**rerank 是可选依赖**：`rerank.enabled=false`、配置缺失或运行期不可用时必须降级为纯 RRF 排序，不得让 `/search` 失败，也不得因此拒绝启动。
6. **禁止过拟合评测数据。** 只允许通用正则、通用字典（月份/日期/数字/代码/问句等）和结构性规则；不得加入只在某个 benchmark/语料格式上有效的词表、格式假设或“看过题目”的特征。新增特征至少在两套语料上做消融，单一家族收益不算。


## 2. 项目是什么

一个本机运行的记忆存储服务：客户端把对话消息 `POST /add` 进来，之后用自然语言 `POST /search` 按 `user_id` 召回相关消息；两个接口都要求 Bearer token，`/health` 不需要。单进程，FastAPI + SQLite，数据落在本地文件；不依赖数据库服务或缓存，embedding 模型服务为必需依赖。

- 监听：`0.0.0.0:8005`
- 字符集：英文与中文都支持；中文检索靠 FTS5 + cutword 分词 + CJK 单字/双字扩展
- 检索：FTS5 + trigram + dense 向量做 RRF 融合，再经 cross-encoder 精排（`bge-reranker-v2-m3`，可选）重排。邻居扩展（`neighbor_message_chars`）默认关闭：它能让 gold 落进邻居文本从而提高 reached@K，但端到端答题反而下降（同字符预算下多给条目优于每条更长），详见 `history.md`。具体口径见本地 `history.md`（不推送）。
