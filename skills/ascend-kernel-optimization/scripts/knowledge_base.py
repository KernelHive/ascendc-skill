"""
RAG Knowledge Base for ascend-kernel-optimization skill.
使用 optimization_points.json 的本地关键词检索，不依赖 embedding。
"""

import json
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set

JSON_FILENAME = "optimization_points.json"

EN_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "with",
    "without", "was", "were", "will", "would", "can", "could", "should", "than",
    "then", "there", "their", "which", "while", "using", "used", "use", "via",
}

SIGNAL_TOKENS = {
    "softmax", "ub", "gm", "mte2", "mte3", "pipeline", "barrier", "waitflag",
    "setflag", "datacopy", "datacopypad", "vector", "tiling", "core", "multicore",
    "cast", "duplicate", "sync", "stall", "latency", "bandwidth", "tail",
}


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("`", " ")
    text = re.sub(r"[_\-/\\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    text = _normalize_text(text)
    en_tokens = re.findall(r"[a-z0-9]+", text)
    cn_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    tokens: List[str] = []

    for token in en_tokens:
        if len(token) <= 1:
            continue
        if token in EN_STOP_WORDS:
            continue
        tokens.append(token)

    tokens.extend(cn_tokens)
    return tokens


class JsonKnowledgeBase:
    """基于 optimization_points.json 的本地检索知识库。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.items: List[Dict[str, Any]] = []
        self._doc_tokens: List[Counter] = []
        self._doc_token_sets: List[Set[str]] = []
        self._doc_search_text: List[str] = []
        self.load()

    def _resolve_json_path(self) -> str:
        if os.path.isfile(self.db_path):
            return self.db_path
        return os.path.join(self.db_path, JSON_FILENAME)

    def _build_doc_index(self):
        self._doc_tokens = []
        self._doc_token_sets = []
        self._doc_search_text = []
        for item in self.items:
            title = str(item.get("title", ""))
            description = str(item.get("description", ""))
            bottleneck = str(item.get("bottleneck", ""))
            search_text = "\n".join([title, description, bottleneck]).strip()
            tokens = _tokenize(search_text)
            token_counter = Counter(tokens)
            self._doc_tokens.append(token_counter)
            self._doc_token_sets.append(set(token_counter.keys()))
            self._doc_search_text.append(_normalize_text(search_text))

    def load(self):
        """从 optimization_points.json 加载并建立索引。"""
        json_path = self._resolve_json_path()
        if not os.path.exists(json_path):
            print(f"[RAG] Knowledge file not found: {json_path}")
            self.items = []
            self._build_doc_index()
            return

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"[RAG] Invalid knowledge file format: expected list, got {type(data)}")
            self.items = []
            self._build_doc_index()
            return

        self.items = [item for item in data if isinstance(item, dict)]
        self._build_doc_index()
        print(f"[RAG] Loaded {len(self.items)} optimization points from: {json_path}")

    def _score(self, query: str, idx: int) -> float:
        query_norm = _normalize_text(query)
        query_tokens = _tokenize(query_norm)
        if not query_tokens:
            return 0.0

        query_set = set(query_tokens)
        doc_counter = self._doc_tokens[idx]
        doc_set = self._doc_token_sets[idx]
        doc_text = self._doc_search_text[idx]

        overlap = query_set.intersection(doc_set)
        overlap_score = 0.0
        for token in overlap:
            weight = 2.0 if token in SIGNAL_TOKENS else 1.0
            overlap_score += weight

        tf_score = 0.0
        for token in query_tokens:
            if token in doc_counter:
                tf_score += min(doc_counter[token], 3) * 0.2

        phrase_score = 0.0
        if query_norm and query_norm in doc_text:
            phrase_score += 2.0

        similarity = SequenceMatcher(None, query_norm[:400], doc_text[:1800]).ratio()
        similarity_score = similarity * 2.0
        return overlap_score + tf_score + phrase_score + similarity_score

    def retrieve(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """根据 bottleneck 文本做本地检索。"""
        if not self.items:
            print("[RAG] Knowledge base is empty, returning empty results")
            return []

        k = max(1, top_k)
        scored = []
        for idx in range(len(self.items)):
            scored.append((self._score(query, idx), idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored[:k]
        results = []
        for score, idx in top_scored:
            results.append({
                "index": idx,
                "score": float(score),
                "content": self.items[idx],
            })
        return results


def create_knowledge_base(db_path: str, api_config: Optional[dict] = None) -> JsonKnowledgeBase:
    """创建本地 JSON 检索知识库。api_config 保留仅为兼容旧调用。"""
    _ = api_config
    return JsonKnowledgeBase(db_path=db_path)
