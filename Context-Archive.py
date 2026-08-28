#!/usr/bin/env python3
"""
Context Archive - 本地语义记忆系统
====================================
为 AI Agent 提供长期记忆能力，支持归档、检索、自动压缩和遗忘。

Copyright (c) 2026 ToskaZhang
Licensed under MIT License
"""

import os
import re
import sys
import json
import math
import time
import pickle
import subprocess
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

# ========== 自动安装依赖 ==========
def _auto_install_deps():
    """自动安装缺失的依赖（受环境变量 AUTO_INSTALL_DEPS 控制）"""
    auto_install = os.environ.get("AUTO_INSTALL_DEPS", "0") == "1"
    if not auto_install:
        return
    
    deps = []
    try:
        import numpy
    except ImportError:
        deps.append("numpy")
    
    try:
        import sklearn
    except ImportError:
        deps.append("scikit-learn")
    
    try:
        import jieba
    except ImportError:
        deps.append("jieba")
    
    if deps:
        print(f"📦 自动安装缺失的依赖: {', '.join(deps)}")
        for dep in deps:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", dep],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"  ✅ {dep} 安装成功")
            except subprocess.CalledProcessError:
                print(f"  ❌ {dep} 安装失败，请手动安装: pip install {dep}")


# 尝试自动安装依赖
_auto_install_deps()

# ========== 导入依赖 ==========
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    print("⚠️  jieba 未安装，将使用简单分词器，中文语义检索效果可能下降。")
    print("💡 运行: pip install jieba 提升效果")
    class FallbackTokenizer:
        @staticmethod
        def lcut(text: str) -> List[str]:
            return [c for c in text if c.strip()]
    jieba = FallbackTokenizer()

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False
    print("❌ NumPy 未安装，这是必需的依赖。")
    print("💡 运行: pip install numpy")
    raise ImportError("NumPy is required. Please install: pip install numpy")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("❌ scikit-learn 未安装，这是必需的依赖。")
    print("💡 运行: pip install scikit-learn")
    raise ImportError("scikit-learn is required. Please install: pip install scikit-learn")


# ========== 数据结构 ==========

@dataclass
class ArchiveEntry:
    role: str = "user"
    content: str = ""
    timestamp: str = ""
    importance: float = 1.0
    session_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_access: Optional[str] = None
    tokens: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ArchiveEntry":
        return cls(**data)


@dataclass
class ArchiverConfig:
    memory_dir: str = "./context-memory"
    max_tokens_per_session: int = 4000
    min_importance: float = 0.1
    importance_time_decay_days: int = 30
    compression_factor: float = 0.7
    max_features: int = 1000
    auto_compress_threshold: float = 0.3


# ========== 核心类 ==========

class ContextArchiver:
    def __init__(self, config: Optional[ArchiverConfig] = None, memory_dir: Optional[str] = None):
        self.config = config or ArchiverConfig()
        if memory_dir:
            self.config.memory_dir = memory_dir
        self.memory_dir = os.path.abspath(self.config.memory_dir)
        self._entries: List[ArchiveEntry] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._vectors: Optional[np.ndarray] = None
        self._is_dirty = False

        self._build_vectorizer()
        os.makedirs(self.memory_dir, exist_ok=True)
        self._load()

    def _build_vectorizer(self):
        tokenizer = jieba.lcut if HAS_JIEBA else None
        self._vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            tokenizer=tokenizer,
            token_pattern=None,
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 2),
        )

    # ---------- 核心 API ----------
    def archive(self, role: str, content: str, session_id: str = "default",
                importance: float = 1.0, metadata: Optional[Dict] = None,
                timestamp: Optional[str] = None) -> ArchiveEntry:
        if not content or not content.strip():
            raise ValueError("记忆内容不能为空")
        if not (0.0 <= importance <= 1.0):
            raise ValueError("重要度必须在 0.0 ~ 1.0 之间")

        tokens = self._estimate_tokens(content)
        entry = ArchiveEntry(
            role=role,
            content=content.strip(),
            timestamp=timestamp or datetime.now().isoformat(),
            importance=importance,
            session_id=session_id,
            metadata=metadata or {},
            tokens=tokens,
        )
        self._entries.append(entry)
        self._is_dirty = True
        self._maybe_compress()
        self._save()
        return entry

    def search(self, query: str, session_id: Optional[str] = None, top_k: int = 5,
               min_importance: Optional[float] = None, recency_weight: float = 0.0) -> List[Dict]:
        if not query or not query.strip() or not self._entries:
            return []

        min_imp = min_importance or self.config.min_importance
        valid_entries = [
            e for e in self._entries
            if e.importance >= min_imp and (session_id is None or e.session_id == session_id)
        ]
        if not valid_entries:
            return []

        self._rebuild_vectors(valid_entries)
        try:
            query_vector = self._vectorizer.transform([query])
        except Exception:
            return []

        if self._vectors is None or self._vectors.shape[0] == 0:
            return []

        similarities = cosine_similarity(query_vector, self._vectors).flatten()

        if recency_weight > 0:
            now = datetime.now()
            for i, entry in enumerate(valid_entries):
                try:
                    ts = datetime.fromisoformat(entry.timestamp)
                    age_days = (now - ts).total_seconds() / 86400
                    recency_score = max(0, 1 - age_days / 365)
                    similarities[i] = (1 - recency_weight) * similarities[i] + recency_weight * recency_score
                except Exception:
                    pass

        indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in indices:
            if similarities[idx] > 0:
                entry = valid_entries[idx]
                result = entry.to_dict()
                result['relevance'] = float(similarities[idx])
                results.append(result)

        # 更新访问计数
        for result in results:
            for entry in self._entries:
                if (entry.content == result['content'] and
                    entry.timestamp == result['timestamp'] and
                    entry.session_id == result['session_id']):
                    entry.access_count += 1
                    entry.last_access = datetime.now().isoformat()
                    break

        if any(r.get('access_count', 0) > 0 for r in results):
            self._is_dirty = True
            self._save()

        return results

    def fetch_relevant(self, query: str, session_id: Optional[str] = None,
                       max_tokens: int = 500, min_importance: Optional[float] = None) -> str:
        results = self.search(query, session_id, top_k=20, min_importance=min_importance)
        if not results:
            return ""

        selected = []
        total_tokens = 0
        for result in results:
            content_tokens = self._estimate_tokens(result['content'])
            if total_tokens + content_tokens <= max_tokens:
                selected.append(result)
                total_tokens += content_tokens
            else:
                remaining = max_tokens - total_tokens
                if remaining > 20:
                    truncated = self._truncate_to_tokens(result['content'], remaining)
                    if truncated:
                        result['content'] = truncated
                        selected.append(result)
                break

        if not selected:
            return ""

        context_parts = []
        for result in selected:
            content = result['content']
            role = result.get('role', 'user')
            prefix = f"[{role}] " if role != 'user' else ""
            context_parts.append(f"{prefix}{content}")
        return "\n".join(context_parts)

    def get_stats(self, session_id: Optional[str] = None) -> Dict:
        entries = self._entries
        if session_id:
            entries = [e for e in entries if e.session_id == session_id]
        if not entries:
            return {"total_entries": 0, "total_tokens": 0, "avg_importance": 0,
                    "total_access_count": 0, "compression_ratio": 1.0, "session_id": session_id}

        total_tokens = sum(e.tokens for e in entries)
        avg_importance = sum(e.importance for e in entries) / len(entries)
        total_access = sum(e.access_count for e in entries)
        valid_count = sum(1 for e in entries if e.importance >= self.config.min_importance)

        return {
            "total_entries": len(entries),
            "total_tokens": total_tokens,
            "avg_importance": avg_importance,
            "total_access_count": total_access,
            "min_importance": self.config.min_importance,
            "valid_entries": valid_count,
            "compression_ratio": valid_count / len(entries) if entries else 1.0,
            "session_id": session_id,
        }

    def list_sessions(self) -> List[str]:
        return sorted({e.session_id for e in self._entries})

    def prune_session(self, session_id: str) -> int:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.session_id != session_id]
        self._is_dirty = True
        self._rebuild_vectors()
        self._save()
        return before - len(self._entries)

    # ---------- 压缩 ----------
    def compress(self, force: bool = False) -> int:
        if not self._entries:
            return 0

        now = datetime.now()
        to_remove = []
        for i, entry in enumerate(self._entries):
            entry_importance = self._compute_entry_importance(entry)
            if entry_importance < self.config.min_importance:
                to_remove.append(i)
                continue
            if entry.access_count == 0:
                try:
                    ts = datetime.fromisoformat(entry.timestamp)
                    age_days = (now - ts).total_seconds() / 86400
                    if age_days > 30:
                        to_remove.append(i)
                except Exception:
                    pass

        if to_remove:
            for idx in sorted(to_remove, reverse=True):
                del self._entries[idx]
            self._is_dirty = True
            self._rebuild_vectors()
            self._save()
        return len(to_remove)

    def _maybe_compress(self):
        if not self._entries:
            return
        valid_count = sum(1 for e in self._entries if e.importance >= self.config.min_importance)
        ratio = valid_count / len(self._entries)
        if ratio < self.config.auto_compress_threshold:
            self.compress()

    def _compute_entry_importance(self, entry: ArchiveEntry) -> float:
        try:
            ts = datetime.fromisoformat(entry.timestamp)
            age_days = (datetime.now() - ts).total_seconds() / 86400
            decay = math.exp(-age_days / self.config.importance_time_decay_days)
        except Exception:
            decay = 1.0
        access_boost = min(0.3, entry.access_count * 0.05)
        return min(1.0, entry.importance * decay + access_boost)

    # ---------- 向量管理 ----------
    def _rebuild_vectors(self, entries: Optional[List[ArchiveEntry]] = None):
        target = entries if entries is not None else self._entries
        if not target:
            self._vectors = None
            return
        contents = [e.content for e in target]
        try:
            self._build_vectorizer()
            self._vectors = self._vectorizer.fit_transform(contents)
        except Exception as e:
            print(f"Warning: Vector rebuild failed: {e}")
            self._vectors = None

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        others = len(re.findall(r'[0-9]', text))
        return int(chinese_chars + english_words * 1.3 + others * 0.5)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        estimated = self._estimate_tokens(text)
        if estimated <= max_tokens:
            return text
        ratio = max_tokens / estimated
        char_limit = int(len(text) * ratio)
        truncated = text[:char_limit]
        for sep in ['。', '！', '？', '.\n', '。\n', '！\n', '？\n']:
            if sep in truncated:
                truncated = truncated.rsplit(sep, 1)[0] + sep
                break
        return truncated.strip()

    # ---------- 持久化 ----------
    def _save(self):
        if not self._is_dirty:
            return
        entries_path = os.path.join(self.memory_dir, "entries.json")
        try:
            with open(entries_path, 'w', encoding='utf-8') as f:
                json.dump([e.to_dict() for e in self._entries], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save entries: {e}")

        vec_path = os.path.join(self.memory_dir, "vectorizer.pkl")
        try:
            with open(vec_path, 'wb') as f:
                pickle.dump(self._vectorizer, f)
        except Exception as e:
            print(f"Warning: Failed to save vectorizer: {e}")
        self._is_dirty = False

    def _load(self):
        entries_path = os.path.join(self.memory_dir, "entries.json")
        if not os.path.exists(entries_path):
            return
        try:
            with open(entries_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._entries = [ArchiveEntry.from_dict(d) for d in data]
        except Exception as e:
            print(f"Warning: Failed to load entries: {e}")
            return

        vec_path = os.path.join(self.memory_dir, "vectorizer.pkl")
        if os.path.exists(vec_path):
            try:
                with open(vec_path, 'rb') as f:
                    self._vectorizer = pickle.load(f)
            except Exception:
                print("Warning: Failed to load vectorizer, rebuilding...")
                self._rebuild_vectors()
        else:
            self._rebuild_vectors()

        if self._vectorizer is None:
            self._rebuild_vectors()

    def clear(self):
        self._entries = []
        self._vectors = None
        self._is_dirty = True
        self._save()

    def export(self) -> List[Dict]:
        return [e.to_dict() for e in self._entries]

    def import_from(self, data: List[Dict]):
        self._entries = [ArchiveEntry.from_dict(d) for d in data]
        self._rebuild_vectors()
        self._is_dirty = True
        self._save()

    def check_health(self) -> Dict:
        return {
            "entries_count": len(self._entries),
            "has_vectorizer": self._vectorizer is not None,
            "has_vectors": self._vectors is not None,
            "memory_dir_exists": os.path.exists(self.memory_dir),
            "memory_dir_writable": os.access(self.memory_dir, os.W_OK),
            "jieba_available": HAS_JIEBA,
            "numpy_available": HAS_NP,
            "sklearn_available": HAS_SKLEARN,
            "auto_install_enabled": os.environ.get("AUTO_INSTALL_DEPS", "0") == "1",
        }


# ========== 顶层 API ==========

_archiver = None

def get_archiver() -> ContextArchiver:
    global _archiver
    if _archiver is None:
        _archiver = ContextArchiver()
    return _archiver

def init(config: Optional[ArchiverConfig] = None, memory_dir: Optional[str] = None) -> ContextArchiver:
    global _archiver
    _archiver = ContextArchiver(config=config, memory_dir=memory_dir)
    return _archiver

# 参数顺序：text 在前，role 可选，更方便
def archive(
    text: str,
    role: str = "user",
    session_id: str = "default",
    importance: float = 1.0
) -> Dict:
    entry = get_archiver().archive(role, text, session_id, importance)
    return entry.to_dict()

def search(
    query: str,
    top_k: int = 5,
    session_id: Optional[str] = None,
    min_importance: Optional[float] = None
) -> List[Dict]:
    return get_archiver().search(query, session_id, top_k, min_importance)

def fetch_relevant(
    query: str,
    session_id: Optional[str] = None,
    max_tokens: int = 500
) -> str:
    return get_archiver().fetch_relevant(query, session_id, max_tokens)

def get_stats(session_id: Optional[str] = None) -> Dict:
    return get_archiver().get_stats(session_id)

def list_sessions() -> List[str]:
    return get_archiver().list_sessions()

def check_health() -> Dict:
    return get_archiver().check_health()


# ========== 命令行入口 ==========

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Context-Archive 命令行工具")
    parser.add_argument("--dir", default="./context-memory", help="记忆存储目录")
    parser.add_argument("--archive", "-a", nargs="+", help="归档内容 (格式: [role:]content)")
    parser.add_argument("--search", "-s", help="搜索记忆")
    parser.add_argument("--top", "-k", type=int, default=5, help="返回条数")
    parser.add_argument("--session", "-S", help="会话 ID")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    parser.add_argument("--clear", action="store_true", help="清空所有记忆")
    parser.add_argument("--export", action="store_true", help="导出所有记忆")
    parser.add_argument("--list-sessions", action="store_true", help="列出所有会话")
    parser.add_argument("--health", action="store_true", help="健康检查")
    parser.add_argument("--auto-install", action="store_true", help="启用自动安装依赖")

    args = parser.parse_args()

    # 如果用户传了 --auto-install，设置环境变量
    if args.auto_install:
        os.environ["AUTO_INSTALL_DEPS"] = "1"
        print("✅ 自动安装依赖已启用")

    archiver = ContextArchiver(memory_dir=args.dir)

    if args.health:
        health = archiver.check_health()
        print("🏥 健康检查报告:")
        for k, v in health.items():
            print(f"  {k}: {v}")
        return

    if args.clear:
        archiver.clear()
        print("✅ 已清空所有记忆")
        return

    if args.archive:
        for item in args.archive:
            if ':' in item:
                role, content = item.split(':', 1)
            else:
                role, content = 'user', item
            entry = archiver.archive(role, content, session_id=args.session or "default")
            print(f"✅ 已归档: [{entry.role}] {entry.content[:50]}... (重要度: {entry.importance})")

    if args.stats:
        stats = archiver.get_stats(session_id=args.session)
        print("\n📊 记忆统计:")
        for k, v in stats.items():
            if k == 'session_id':
                continue
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

    if args.search:
        print(f"\n🔍 搜索: '{args.search}'")
        results = archiver.search(args.search, session_id=args.session, top_k=args.top)
        if results:
            for i, r in enumerate(results, 1):
                print(f"\n{i}. [相关度: {r.get('relevance', 0):.2f}] {r['content']}")
                print(f"   角色: {r['role']} | 重要度: {r['importance']:.2f} | 访问: {r['access_count']}次")
        else:
            print("  无匹配结果")

    if args.export:
        data = archiver.export()
        print(json.dumps(data, ensure_ascii=False, indent=2))

    if args.list_sessions:
        sessions = archiver.list_sessions()
        print("📋 会话列表:")
        for s in sessions:
            count = len([e for e in archiver._entries if e.session_id == s])
            print(f"  {s}: {count} 条记忆")


if __name__ == "__main__":
    main()
