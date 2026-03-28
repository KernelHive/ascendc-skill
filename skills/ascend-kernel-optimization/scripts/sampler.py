"""
Kernel Optimizer Sampler for ascend-kernel-optimization skill.
负责:
1. 识别bottleneck
2. 检索RAG知识库
3. 生成代码修改建议
"""

import json
import os
import random
import re
import copy
from typing import Dict, List, Optional, Tuple, Any
from openai import OpenAI


class Sampler:
    """生成候选代码的采样器"""

    def __init__(
        self,
        evaluator,
        llm_config,
        log_dir: str,
        op_name: str,
        op_category: str,
        op_code: Dict[str, Dict[str, str]],
        knowledge_base=None,
    ):
        self.evaluator = evaluator
        self.llm_config = llm_config
        self.log_dir = log_dir
        self.op_name = op_name
        self.op_category = op_category
        self.op_code = op_code  # {"subdir": {"filename": "content"}}
        self.knowledge_base = knowledge_base
        
        # 记录历史bottleneck
        self.history_bottleneck = []

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用LLM API"""
        if not self.llm_config.api_key:
            print("[Warning] LLM API key not set, using fallback")
            return ""
        
        try:
            client = OpenAI(
                base_url=self.llm_config.base_url, 
                api_key=self.llm_config.api_key
            )
            response = client.chat.completions.create(
                model=self.llm_config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                n=1,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"[LLM Error] {e}")
            return ""

    def get_bottleneck(self) -> str:
        """
        识别当前代码的性能瓶颈
        
        Returns:
            str: 瓶颈描述
        """
        system_prompt = """You are an expert AscendC operator performance tuning engineer.
Your task is to identify the most critical performance bottleneck in the given code.

IMPORTANT: 
- Provide analysis in a single, concise paragraph (max 3 sentences)
- Do NOT include any introductory remarks, headers, or conversational filler.
- Focus on specific code patterns that cause performance issues"""

        # 组织代码文本
        code_str = ""
        for subdir, files in self.op_code.items():
            for filename, content in files.items():
                code_str += f"\n--- FILE: {subdir}/{filename} ---\n{content}\n"

        prompt = f"""
Analyze the following AscendC operator code:
- Name: {self.op_name}
- Category: {self.op_category}
- Code: 
{code_str}

Context:
- Previously identified bottlenecks: {self.history_bottleneck}

Based on the code structure, memory access patterns, and instruction scheduling, identify the single most impactful performance bottleneck remaining that has NOT been mentioned in the history above.
Output ONLY the description of this new bottleneck in one concise paragraph.
"""
        
        bottleneck = self._call_llm(system_prompt, prompt)
        
        if bottleneck:
            self.history_bottleneck.append(bottleneck)
            print(f"[Bottleneck Identified]: {bottleneck}")
        
        return bottleneck or "Unknown bottleneck"

    def retrieve_experience(self, bottleneck: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        从RAG知识库检索相关优化经验
        
        Args:
            bottleneck: 瓶颈描述
            top_k: 检索数量
            
        Returns:
            List[Dict]: 相关经验列表
        """
        if not self.knowledge_base:
            print("[RAG] Knowledge base not available")
            return []
        
        results = self.knowledge_base.retrieve(bottleneck, top_k=top_k)
        print(f"[RAG] Retrieved {len(results)} experiences for bottleneck")
        
        return results

    def generate_modification(
        self, 
        bottleneck: str, 
        experiences: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, str]]:
        """
        根据bottleneck和经验生成代码修改
        
        Args:
            bottleneck: 瓶颈描述
            experiences: 检索到的经验
            
        Returns:
            (modification_reason, modified_codes)
            - modification_reason: 修改说明
            - modified_codes: 修改后的文件 {"subdir/filename": "content"}
        """
        system_prompt = """You are an expert AscendC operator optimization engineer.
Your task is to modify the operator code to address the identified bottleneck.
Output ONLY the diff in the specified format.
CONSTRAINTS:
- Do NOT wrap code in ```cpp tags, use the diff format below
- Keep the modification minimal and focused
- If no modification is needed, say "NO_CHANGE_NEEDED"
"""
        
        # 组织经验内容
        experiences_text = ""
        for i, exp in enumerate(experiences):
            content = exp.get("content", {})
            if isinstance(content, dict):
                title = content.get("title", "")
                description = content.get("description", "")
                experience_bottleneck = content.get("bottleneck", "")
                code_diff = content.get("code_diff", "")
                experiences_text += f"""
---
Experience {i+1}:
Title: {title}
Description: {description}
Matched Bottleneck: {experience_bottleneck}
Reference Diff: {code_diff}
"""
        
        # 组织代码
        code_str = ""
        for subdir, files in self.op_code.items():
            for filename, content in files.items():
                code_str += f"\n--- FILE: {subdir}/{filename} ---\n{content}\n"

        prompt = f"""
Current Bottleneck: {bottleneck}

Relevant Optimization Experiences:
{experiences_text}

Original Code:
{code_str}

Please generate code modifications to address the bottleneck.
Output format (diff):
<<<< op_kernel/kernel_name.cpp
OLD_CODE
====
NEW_CODE
>>>>

If no modification is needed, output: NO_CHANGE_NEEDED
"""
        
        diff_result = self._call_llm(system_prompt, prompt)
        
        # 解析diff
        modified_codes = self._parse_diff(diff_result)
        
        return diff_result, modified_codes

    def _parse_diff(self, diff_content: str) -> Dict[str, str]:
        """解析diff内容，生成修改后的代码"""
        if not diff_content or diff_content.strip() == "NO_CHANGE_NEEDED":
            return {}
        
        original_codes = copy.deepcopy(self.op_code)
        
        # 辅助函数：标准化行（用于模糊匹配）
        def _normalize_lines_for_fuzzy(text: str) -> List[str]:
            lines = text.split('\n')
            normalized = []
            for line in lines:
                # 移除多余空白
                stripped = re.sub(r'\s+', ' ', line.strip())
                if stripped:
                    normalized.append(stripped)
            return normalized

        # 查找唯一匹配
        def _find_unique_exact(haystack: str, needle: str) -> Optional[int]:
            if not needle:
                return None
            first = haystack.find(needle)
            if first < 0:
                return None
            # 确保只有一处匹配
            if haystack.find(needle, first + 1) >= 0:
                return None
            return first

        def _find_unique_by_line_window(code_lines: List[str], old_lines: List[str]) -> Optional[int]:
            if not old_lines:
                return None
            n = len(old_lines)
            if n > len(code_lines):
                return None
            matches = []
            for i in range(len(code_lines) - n + 1):
                if code_lines[i:i+n] == old_lines:
                    matches.append(i)
                    if len(matches) > 1:
                        return None
            return matches[0] if matches else None

        def _apply_single_text_diff(text: str, old_raw: str, new_raw: str) -> str:
            code = text.replace("\r\n", "\n")
            if old_raw.startswith('\n'):
                old_raw = old_raw[1:]
            if new_raw.startswith('\n'):
                new_raw = new_raw[1:]
            if not old_raw and not new_raw:
                return code

            idx = _find_unique_exact(code, old_raw)
            if idx is not None:
                return code[:idx] + new_raw + code[idx + len(old_raw):]

            code_norm_lines = _normalize_lines_for_fuzzy(code)
            old_norm_lines = _normalize_lines_for_fuzzy(old_raw)
            start_line = _find_unique_by_line_window(code_norm_lines, old_norm_lines)
            if start_line is not None:
                new_norm_lines = _normalize_lines_for_fuzzy(new_raw)
                return "\n".join(code_norm_lines[:start_line] + new_norm_lines + code_norm_lines[start_line + len(old_norm_lines):])
            return code

        # 多文件字典模式
        new_codes = copy.deepcopy(original_codes)
        path_to_keys = {}
        for sub_dir, files in original_codes.items():
            for filename in files:
                path_to_keys[os.path.join(sub_dir, filename)] = (sub_dir, filename)

        # 解析diff块
        blocks = re.findall(
            r'<<<<\s*(.*?)\s*\n?(.*?)\s*====\s*(.*?)\s*>>>>', 
            diff_content.replace("\r\n", "\n"), 
            re.DOTALL
        )
        
        for block_str in blocks:
            if len(block_str) != 3:
                continue
            
            file_path, old_raw, new_raw = block_str
            
            # 查找目标文件
            target_keys = path_to_keys.get(file_path.strip())
            if not target_keys:
                # 模糊匹配
                for p, k in path_to_keys.items():
                    if file_path.strip() in p or p in file_path.strip():
                        target_keys = k
                        break
            
            if target_keys:
                subdir, filename = target_keys
                new_codes[subdir][filename] = _apply_single_text_diff(
                    new_codes[subdir][filename], 
                    old_raw, 
                    new_raw
                )
        
        return new_codes

    def sample(self, iteration: int) -> Tuple[Optional[str], Optional[Dict[str, str]], float]:
        """
        执行一轮采样: 识别bottleneck -> 检索RAG -> 生成修改
        
        Args:
            iteration: 当前迭代轮次
            
        Returns:
            (modification_reason, modified_codes, score)
        """
        # 1. 识别bottleneck
        bottleneck = self.get_bottleneck()
        
        # 2. 检索RAG
        experiences = self.retrieve_experience(bottleneck, top_k=1)
        
        # 3. 生成代码修改
        modification_reason, modified_codes = self.generate_modification(bottleneck, experiences)
        
        if not modified_codes:
            print(f"[Iteration {iteration}] No modification generated")
            return None, None, -10000
        
        # 4. 评估
        success, score = self.evaluator.evaluate(
            modified_codes=modified_codes,
            log_dir=self.log_dir,
            island_id=0,
            iteration=iteration
        )
        
        print(f"[Iteration {iteration}] Score: {score}, Success: {success}")
        
        return modification_reason, modified_codes, score
