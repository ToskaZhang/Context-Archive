# Context-Archive-
一个零外部 API、基于 TF‑IDF + Jieba 分词 的本地上下文归档与检索工具，用 纯 JSON 存储，支持自动压缩以应对大规模记录
🌟 特性
✅ 纯本地 – 所有数据保留在您的磁盘上，无隐私泄露风险

✅ 中文优化 – 内置 Jieba 分词，对中文文本检索效果更佳

✅ 语义检索 – 基于 TF‑IDF 向量空间模型，实现模糊语义匹配

✅ 自动压缩 – 当条目超过阈值时，根据“信息量 + 时间衰减”自动修剪，保持系统轻盈

✅ 零依赖服务 – 仅需 Python 3.6+ 及几个常用库（jieba, numpy, scikit-learn）

✅ 即拿即用 – 单文件设计，拷贝即可集成到您的项目中
📦 安装
环境要求
Python 3.6 或更高版本

pip（Python 包管理器）

1. 克隆仓库
bash
git clone https://github.com/yourname/context-archive.git
cd context-archive
2. 安装依赖
bash
pip install jieba numpy scikit-learn
或者使用提供的 requirements.txt（若存在）：

bash
pip install -r requirements.txt
3. 运行示例（测试安装）
bash
python context_archiver.py
您将看到归档示例消息、执行搜索并输出状态。

🚀 快速开始
python
from context_archiver import archive, search, init

# 初始化（会自动创建目录和向量器）
init()

# 归档消息
archive("user", "如何解决 Docker 在 Windows 上的权限问题？", session_id="tech-support")
archive("assistant", "将当前用户添加到 docker-users 组，然后重启系统。", session_id="tech-support")

# 搜索相关历史
results = search("Docker 权限修复", top_k=2)
for r in results:
    print(f"[{r['role']}] {r['text']} (score: {r['score']})")
⚙️ 配置
您可以通过 ArchiverConfig 自定义行为：

python
from context_archiver import ArchiverConfig, ContextArchiver

config = ArchiverConfig(
    memory_dir="./my_memory",          # 存储目录
    max_chunk_size=300,                # 文本分块大小
    retrain_interval=50,               # 每 50 条归档重训一次向量器
    max_entries_before_compress=5000,  # 超过 5000 条触发压缩
    compress_keep_ratio=0.7,           # 压缩后保留 70% 条目
)
archiver = ContextArchiver(config)
📖 API 概览
函数	说明
archive(role, text, session_id, timestamp)	归档一条消息，自动分块并存储向量
search(query, top_k, session_id)	检索与查询最相似的 top_k 条记录
fetch_relevant(context, max_tokens, session_id)	返回格式化后的上下文文本（用于 LLM 提示）
compress_context(max_entries)	手动触发压缩（自动也会触发）
retrain()	手动重训向量器（更新所有向量）
cleanup_old_entries(days)	删除指定天数前的旧记录
get_status()	获取统计信息
🔄 工作流程可视化
以下是用 Mermaid 绘制的核心流程图，您可以在 GitHub README 中渲染。
graph TD
    A[用户输入] -->|归档| B(文本分块)
    B --> C[Jieba 分词]
    C --> D[TF-IDF 向量化]
    D --> E[存储 JSON<br/>（含向量与元数据）]
    E --> F{条目数超阈值?}
    F -->|是| G[压缩流程]
    F -->|否| H[结束归档]

    G --> I[计算每条记录的重要度<br/>= 向量权重和 × 时间衰减]
    I --> J[按重要度排序]
    J --> K[保留前 N 条]
    K --> L[重训向量器]
    L --> M[更新存储向量]

    N[用户查询] -->|检索| O[查询向量化]
    O --> P[计算余弦相似度]
    P --> Q[排序返回 Top-K]

    style A fill:#f9f,stroke:#333
    style N fill:#f9f,stroke:#333
    style G fill:#bbf,stroke:#333
    🗂️ 目录结构（运行后自动生成）
text
E:\context-memory/                 # 默认存储目录
├── entries.json                   # 所有记忆条目（含向量）
├── metadata.json                  # 统计信息
├── vectorizer.json                # TF-IDF 词汇表与 IDF
└── sessions/                      # （预留）会话级隔离
🤝 贡献
欢迎提交 Issue 和 Pull Request。
若您需要新增功能（如支持更多 Embedding 模型、接入向量数据库等），请先通过 Issue 讨论。

🙏 致谢
Jieba – 中文分词

scikit-learn – TF-IDF 与相似度计算

现在就为您的 AI 应用赋予长期记忆吧！ 🧠
如果觉得有用，别忘了点 ⭐ 支持一下～
