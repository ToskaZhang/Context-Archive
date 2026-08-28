#!/usr/bin/env python3
"""
Context Archive - 本地语义记忆系统
====================================
为 AI Agent 提供长期记忆能力，支持归档、检索、自动压缩和遗忘。

Copyright (c) 2026 ToskaZhang
Licensed under MIT License

Features:
- 本地存储，零外部依赖（可选）
- 中文分词支持（Jieba）
- TF-IDF + 余弦相似度语义检索
- 自动压缩与遗忘（基于信息量 + 时间衰减）
- 完全跨平台路径支持
- 统计信息查询
"""

import os
import re
import sys
import json
import math
import time
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

# ========== 可选依赖 ==========
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    # 提供一个简单的 fallback 分词器
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
    raise ImportError("NumPy is required for Context-Archive. Please install: pip install numpy")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    raise ImportError("scikit-learn is required for Context-Archive. Please install: pip install scikit-learn")


# ========== 数据结构 ==========

@dataclass
class ArchiveEntry:
    """单条记忆条目"""
    content: str
    timestamp: str
    importance: float = 1.0
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
class ArchiveConfig:
    """归档配置"""
    max_tokens: int = 4000
    min_importance: float = 0.1
    importance_time_decay_days: int = 30
    compression_factor: float = 0.7
    max_features: int = 1000
    auto_compress_threshold: float = 0.3  # 当有效条目占比低于此值时触发压缩


# ========== 核心类 ==========

class ContextArchive:
    """
    本地语义记忆系统
    
    使用方式:
        archive = ContextArchive()
        archive.archive("用户说他的项目用 Python 3.11")
        archive.archive("用户喜欢用 FastAPI")
        
        results = archive.search("用户用什么语言?")
        print(results)  # 返回最相关的记忆
        
        # 获取结构化上下文
        context = archive.fetch_relevant("Python 项目", max_tokens=500)
        print(context)
    """
    
    def __init__(
        self,
        memory_dir: Optional[str] = None,
        config: Optional[ArchiveConfig] = None,
        auto_compress: bool = True
    ):
        """
        初始化记忆系统
        
        Args:
            memory_dir: 记忆存储目录（默认：当前目录下的 context-memory）
            config: 配置对象（默认使用 ArchiveConfig）
            auto_compress: 是否在每次归档后自动检查并压缩
        """
        # ===== 修复 1: 平台无关的默认路径 =====
        if memory_dir is None:
            memory_dir = os.path.join(os.getcwd(), "context-memory")
        self.memory_dir = os.path.abspath(memory_dir)
        
        self.config = config or ArchiveConfig()
        self._entries: List[ArchiveEntry] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._vectors: Optional[np.ndarray] = None
        self._is_dirty = False
        self._auto_compress = auto_compress
        
        # ===== 修复 2: 修正 tokenizer 参数 =====
        # 使用 jieba.lcut 作为分词器，如果可用
        tokenizer = jieba.lcut if HAS_JIEBA else None
        
        self._vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            tokenizer=tokenizer,
            token_pattern=None,  # 使用自定义 tokenizer 时此参数无效
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 2),
        )
        
        # 创建存储目录
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # 加载已有记忆
        self._load()
        
    # ========== 核心 API ==========
    
    def archive(
        self,
        content: str,
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ) -> ArchiveEntry:
        """
        归档一条新记忆
        
        Args:
            content: 记忆内容
            importance: 重要度 (0.0 ~ 1.0)
            metadata: 附加元数据
            timestamp: 时间戳（默认为当前时间）
            
        Returns:
            创建的 ArchiveEntry 对象
        """
        if not content or not content.strip():
            raise ValueError("记忆内容不能为空")
        
        if not (0.0 <= importance <= 1.0):
            raise ValueError("重要度必须在 0.0 ~ 1.0 之间")
        
        # 计算 token 数（估算）
        tokens = self._estimate_tokens(content)
        
        entry = ArchiveEntry(
            content=content.strip(),
            timestamp=timestamp or datetime.now().isoformat(),
            importance=importance,
            metadata=metadata or {},
            tokens=tokens,
        )
        
        self._entries.append(entry)
        self._is_dirty = True
        
        # 自动压缩检查
        if self._auto_compress:
            self._maybe_compress()
        
        self._save()
        return entry
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_importance: Optional[float] = None,
        recency_weight: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        语义搜索记忆
        
        Args:
            query: 查询文本
            top_k: 返回条数
            min_importance: 最低重要度过滤
            recency_weight: 时效性权重 (0.0 ~ 1.0)
            
        Returns:
            匹配的条目列表（按相关度降序）
        """
        if not query or not query.strip():
            return []
        
        if not self._entries:
            return []
        
        # 过滤低重要度条目
        min_imp = min_importance or self.config.min_importance
        valid_entries = [
            e for e in self._entries
            if e.importance >= min_imp
        ]
        
        if not valid_entries:
            return []
        
        # 重建向量（如果需要）
        self._rebuild_vectors(valid_entries)
        
        # 向量化查询
        try:
            query_vector = self._vectorizer.transform([query])
        except Exception:
            # 如果向量器未训练，返回空
            return []
        
        # 计算相似度
        if self._vectors is None or self._vectors.shape[0] == 0:
            return []
        
        similarities = cosine_similarity(query_vector, self._vectors).flatten()
        
        # 应用时效性权重
        if recency_weight > 0:
            now = datetime.now()
            for i, entry in enumerate(valid_entries):
                try:
                    ts = datetime.fromisoformat(entry.timestamp)
                    age_days = (now - ts).total_seconds() / 86400
                    recency_score = max(0, 1 - age_days / 365)  # 1年内线性衰减
                    similarities[i] = (1 - recency_weight) * similarities[i] + recency_weight * recency_score
                except Exception:
                    pass
        
        # 排序并返回
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
            entry_id = self._entries.index(next(e for e in self._entries if e.content == result['content']))
            self._entries[entry_id].access_count += 1
            self._entries[entry_id].last_access = datetime.now().isoformat()
        
        if any(r.get('access_count', 0) > 0 for r in results):
            self._is_dirty = True
            self._save()
        
        return results
    
    def fetch_relevant(
        self,
        query: str,
        max_tokens: int = 500,
        min_importance: Optional[float] = None
    ) -> str:
        """
        获取相关的上下文文本（用于 AI Prompt）
        
        Args:
            query: 查询文本
            max_tokens: 最大 token 数
            min_importance: 最低重要度
            
        Returns:
            组合后的上下文字符串
        """
        results = self.search(query, top_k=20, min_importance=min_importance)
        
        if not results:
            return ""
        
        selected = []
        total_tokens = 0
        
        for result in results:
            # 估算 token 数
            content_tokens = self._estimate_tokens(result['content'])
            if total_tokens + content_tokens <= max_tokens:
                selected.append(result)
                total_tokens += content_tokens
            else:
                # 尝试截断
                remaining = max_tokens - total_tokens
                if remaining > 20:
                    truncated = self._truncate_to_tokens(result['content'], remaining)
                    if truncated:
                        result['content'] = truncated
                        selected.append(result)
                break
        
        if not selected:
            return ""
        
        # 构建上下文字符串
        context_parts = []
        for i, result in enumerate(selected, 1):
            relevance = result.get('relevance', 0)
            importance = result.get('importance', 0)
            content = result['content']
            
            # 如果重要度或相关度较高，加标记
            prefix = ""
            if importance > 0.8 and relevance > 0.3:
                prefix = "⭐ "
            elif importance > 0.5:
                prefix = "▪ "
            
            context_parts.append(f"{prefix}{content}")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息
        
        Returns:
            统计字典
        """
        if not self._entries:
            return {
                "total_entries": 0,
                "total_tokens": 0,
                "avg_importance": 0,
                "total_access_count": 0,
                "compression_ratio": 1.0,
            }
        
        total_tokens = sum(e.tokens for e in self._entries)
        avg_importance = sum(e.importance for e in self._entries) / len(self._entries)
        total_access = sum(e.access_count for e in self._entries)
        
        # 有效条目（重要度 > min_importance）
        valid_count = sum(1 for e in self._entries if e.importance >= self.config.min_importance)
        
        return {
            "total_entries": len(self._entries),
            "total_tokens": total_tokens,
            "avg_importance": avg_importance,
            "total_access_count": total_access,
            "min_importance": self.config.min_importance,
            "valid_entries": valid_count,
            "compression_ratio": valid_count / len(self._entries) if self._entries else 1.0,
        }
    
    # ========== 压缩与维护 ==========
    
    def compress(self, force: bool = False) -> int:
        """
        压缩记忆库：删除低重要度、久未访问的条目
        
        Args:
            force: 是否强制压缩（忽略自动阈值）
            
        Returns:
            删除的条目数
        """
        if not self._entries:
            return 0
        
        now = datetime.now()
        to_remove = []
        
        for i, entry in enumerate(self._entries):
            # 计算综合重要性衰减
            entry_importance = self._compute_entry_importance(entry)
            
            # 如果重要度低于阈值，标记删除
            if entry_importance < self.config.min_importance:
                to_remove.append(i)
                continue
            
            # 如果条目从未被访问且超过 7 天，也考虑删除
            if entry.access_count == 0:
                try:
                    ts = datetime.fromisoformat(entry.timestamp)
                    age_days = (now - ts).total_seconds() / 86400
                    if age_days > 30:
                        to_remove.append(i)
                        continue
                except Exception:
                    pass
        
        # 执行删除（从后往前删除，避免索引错乱）
        if to_remove:
            for idx in sorted(to_remove, reverse=True):
                del self._entries[idx]
            self._is_dirty = True
            self._rebuild_vectors()
            self._save()
        
        return len(to_remove)
    
    def _maybe_compress(self):
        """根据阈值自动触发压缩"""
        if not self._entries:
            return
        
        valid_count = sum(1 for e in self._entries if e.importance >= self.config.min_importance)
        ratio = valid_count / len(self._entries) if self._entries else 1.0
        
        if ratio < self.config.auto_compress_threshold:
            self.compress()
    
    def _compute_entry_importance(self, entry: ArchiveEntry) -> float:
        """计算条目当前重要度（包含时间衰减）"""
        # ===== 修复 4: 指数衰减 =====
        try:
            ts = datetime.fromisoformat(entry.timestamp)
            age_days = (datetime.now() - ts).total_seconds() / 86400
            # 指数衰减：重要度随时间平滑下降
            decay = math.exp(-age_days / self.config.importance_time_decay_days)
        except Exception:
            decay = 1.0
        
        # 访问次数加分
        access_boost = min(0.3, entry.access_count * 0.05)
        
        # 基础重要度 × 时间衰减 + 访问加成
        raw_importance = entry.importance * decay + access_boost
        
        return min(1.0, raw_importance)
    
    # ========== 向量管理 ==========
    
    def _rebuild_vectors(self, entries: Optional[List[ArchiveEntry]] = None):
        """重建 TF-IDF 向量"""
        target = entries if entries is not None else self._entries
        if not target:
            self._vectors = None
            return
        
        # 获取所有内容
        contents = [e.content for e in target]
        
        # 拟合和转换
        try:
            self._vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                tokenizer=jieba.lcut if HAS_JIEBA else None,
                token_pattern=None,
                lowercase=True,
                stop_words=None,
                ngram_range=(1, 2),
            )
            self._vectors = self._vectorizer.fit_transform(contents)
        except Exception as e:
            print(f"Warning: Vector rebuild failed: {e}")
            self._vectors = None
    
    def _estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数（中英文混合）"""
        # 中文按字，英文/数字按词
        if not text:
            return 0
        
        # 中文字符计数
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 英文词计数
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        # 数字/符号计数
        others = len(re.findall(r'[0-9]', text))
        
        # 粗略估算：中文 1 字 ≈ 1 token，英文 1 词 ≈ 1.3 tokens
        return int(chinese_chars + english_words * 1.3 + others * 0.5)
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """按 token 数截断文本"""
        if max_tokens <= 0:
            return ""
        
        estimated = self._estimate_tokens(text)
        if estimated <= max_tokens:
            return text
        
        # 简单截断：按比例
        ratio = max_tokens / estimated
        char_limit = int(len(text) * ratio)
        truncated = text[:char_limit]
        
        # 尝试在句号处截断
        for sep in ['。', '！', '？', '.\n', '。\n', '！\n', '？\n']:
            if sep in truncated:
                truncated = truncated.rsplit(sep, 1)[0] + sep
                break
        
        return truncated.strip()
    
    # ========== 持久化 ==========
    
    def _save(self):
        """保存记忆到磁盘"""
        if not self._is_dirty:
            return
        
        # 保存条目
        entries_path = os.path.join(self.memory_dir, "entries.json")
        try:
            with open(entries_path, 'w', encoding='utf-8') as f:
                json.dump(
                    [e.to_dict() for e in self._entries],
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        except Exception as e:
            print(f"Warning: Failed to save entries: {e}")
        
        # 保存向量器
        if self._vectorizer is not None:
            vec_path = os.path.join(self.memory_dir, "vectorizer.pkl")
            try:
                with open(vec_path, 'wb') as f:
                    pickle.dump(self._vectorizer, f)
            except Exception as e:
                print(f"Warning: Failed to save vectorizer: {e}")
        
        self._is_dirty = False
    
    def _load(self):
        """从磁盘加载记忆"""
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
        
        # 加载向量器
        vec_path = os.path.join(self.memory_dir, "vectorizer.pkl")
        if os.path.exists(vec_path):
            try:
                with open(vec_path, 'rb') as f:
                    self._vectorizer = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load vectorizer: {e}")
                self._rebuild_vectors()
        
        # 如果向量器为空或条目变了，重建
        if self._vectorizer is None or len(self._entries) > 0:
            self._rebuild_vectors()
    
    def clear(self):
        """清空所有记忆"""
        self._entries = []
        self._vectors = None
        self._is_dirty = True
        self._save()
    
    def export(self) -> List[Dict[str, Any]]:
        """导出所有记忆为字典列表"""
        return [e.to_dict() for e in self._entries]
    
    def import_from(self, data: List[Dict[str, Any]]):
        """从字典列表导入记忆"""
        self._entries = [ArchiveEntry.from_dict(d) for d in data]
        self._rebuild_vectors()
        self._is_dirty = True
        self._save()


# ========== 命令行入口 ==========

def main():
    """命令行测试入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Context-Archive 命令行工具")
    parser.add_argument("--dir", default="./context-memory", help="记忆存储目录")
    parser.add_argument("--archive", "-a", nargs="+", help="归档内容")
    parser.add_argument("--search", "-s", help="搜索记忆")
    parser.add_argument("--top", "-k", type=int, default=5, help="返回条数")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    parser.add_argument("--clear", action="store_true", help="清空所有记忆")
    parser.add_argument("--export", action="store_true", help="导出所有记忆")
    
    args = parser.parse_args()
    
    archive = ContextArchive(memory_dir=args.dir)
    
    if args.clear:
        archive.clear()
        print("✅ 已清空所有记忆")
        return
    
    if args.archive:
        content = " ".join(args.archive)
        entry = archive.archive(content)
        print(f"✅ 已归档: {content[:50]}... (重要度: {entry.importance})")
    
    if args.stats:
        stats = archive.get_stats()
        print("\n📊 记忆统计:")
        print(f"  总条目: {stats['total_entries']}")
        print(f"  总 Token: {stats['total_tokens']}")
        print(f"  有效条目: {stats['valid_entries']}")
        print(f"  平均重要度: {stats['avg_importance']:.2f}")
        print(f"  总访问次数: {stats['total_access_count']}")
        print(f"  压缩率: {stats['compression_ratio']:.1%}")
    
    if args.search:
        print(f"\n🔍 搜索: '{args.search}'")
        results = archive.search(args.search, top_k=args.top)
        if results:
            for i, r in enumerate(results, 1):
                print(f"\n{i}. [相关度: {r.get('relevance', 0):.2f}] {r['content']}")
                print(f"   重要度: {r['importance']:.2f} | 访问: {r['access_count']}次")
        else:
            print("  无匹配结果")
    
    if args.export:
        data = archive.export()
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
