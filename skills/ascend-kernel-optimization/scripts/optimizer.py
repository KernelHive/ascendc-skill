"""
Kernel Optimizer for ascend-kernel-optimization skill.
主优化器，协调采样器和评估器完成 kernel 代码优化。
"""

import os
import shutil
from datetime import datetime
from typing import Dict, Optional, Tuple

from .config import get_optimizer_config, get_path_config, get_llm_config
from .evaluator import KernelEvaluator
from .knowledge_base import create_knowledge_base
from .sampler import Sampler


class KernelOptimizer:
    """Kernel 代码优化器"""

    def __init__(
        self,
        op_name: str,
        op_category: str,
        tiling_resource_base: Optional[str] = None,
    ):
        self.op_name = op_name
        self.op_category = op_category
        self.tiling_resource_base = tiling_resource_base

        # 获取配置
        path_config = get_path_config()
        optimizer_config = get_optimizer_config()

        # 设置日志目录
        self.log_dir = os.path.join(path_config.log_base, op_category, op_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化评估器
        self.evaluator = KernelEvaluator(
            log_dir=self.log_dir,
            op_name=op_name,
            op_category=op_category,
            tiling_resource_base=tiling_resource_base,
        )

        # 初始化知识库
        self.knowledge_base = None
        kb_path = path_config.knowledge_base_path
        if os.path.exists(kb_path):
            self.knowledge_base = create_knowledge_base(
                db_path=kb_path
            )
            print(f"[Optimizer] Knowledge base loaded from: {kb_path}")
        else:
            print(f"[Warning] Knowledge base not found at: {kb_path}")

        # 优化配置
        self.max_iterations = optimizer_config.max_iterations

        # 优化状态
        self.op_code = None
        self.best_score = float('inf')
        self.best_code = None

    def _build_unique_output_dir(self) -> str:
        """构建唯一输出目录，避免覆盖历史优化结果"""
        path_config = get_path_config()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_id = f"run_{timestamp}"
        run_root = os.path.join(path_config.output_base, run_id)
        suffix = 1
        while os.path.exists(run_root):
            run_root = os.path.join(path_config.output_base, f"{run_id}_{suffix}")
            suffix += 1

        return os.path.join(run_root, self.op_category, self.op_name)

    def _load_op_code(self) -> Dict[str, Dict[str, str]]:
        """加载算子代码"""
        if self.evaluator.base_op_file_list:
            # 转换为 {"subdir": {"filename": "content"}} 格式
            result = {}
            for rel_path, content in self.evaluator.base_op_file_list.items():
                parts = rel_path.split('/')
                if len(parts) >= 2:
                    subdir = parts[0]
                    filename = '/'.join(parts[1:])
                else:
                    subdir = 'op_kernel'
                    filename = rel_path
                
                if subdir not in result:
                    result[subdir] = {}
                result[subdir][filename] = content
            return result
        else:
            # 从文件系统加载
            base_path = os.path.join(self.tiling_resource_base, self.op_category, self.op_name)
            result = {}
            
            for subdir in ["op_host", "op_kernel"]:
                dir_path = os.path.join(base_path, subdir)
                if not os.path.exists(dir_path):
                    continue
                
                result[subdir] = {}
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, dir_path)
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                result[subdir][rel_path] = f.read()
                        except:
                            pass
            
            return result

    def _save_optimized_code(self, modified_codes: Dict[str, Dict[str, str]]) -> str:
        """保存优化后的代码到输出目录"""
        output_dir = self._build_unique_output_dir()

        # 复制原始算子目录
        source_dir = os.path.join(self.tiling_resource_base, self.op_category, self.op_name)
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source operator directory not found: {source_dir}")
        shutil.copytree(source_dir, output_dir)

        # 应用修改
        for subdir, files in modified_codes.items():
            for filename, content in files.items():
                target_file = os.path.join(output_dir, subdir, filename)
                if os.path.exists(target_file):
                    with open(target_file, "w", encoding="utf-8") as f:
                        f.write(content)

        return output_dir

    def optimize(self) -> Tuple[Optional[Dict[str, Dict[str, str]]], float, str]:
        """执行优化"""
        print(f"Starting kernel optimization for {self.op_category}/{self.op_name}")
        print(f"Max iterations: {self.max_iterations}")

        # 加载算子代码
        self.op_code = self._load_op_code()
        print(f"[Optimizer] Loaded code for {len(self.op_code)} subdirectories")

        # 初始化采样器
        llm_config = get_llm_config()
        sampler = Sampler(
            evaluator=self.evaluator,
            llm_config=llm_config,
            log_dir=self.log_dir,
            op_name=self.op_name,
            op_category=self.op_category,
            op_code=self.op_code,
            knowledge_base=self.knowledge_base,
        )

        # 迭代优化
        for iteration in range(self.max_iterations):
            print(f"\n=== Iteration {iteration + 1}/{self.max_iterations} ===")

            # 执行采样
            modification_reason, modified_codes, score = sampler.sample(iteration)

            if modified_codes and score > -10000:
                # 更新最佳解
                if score < self.best_score:
                    self.best_score = score
                    self.best_code = modified_codes
                    print(f"New best score: {self.best_score}")

        # 保存结果
        if self.best_code:
            output_dir = self._save_optimized_code(self.best_code)
            print(f"\nOptimization complete!")
            print(f"Best score: {self.best_score}")
            print(f"Output saved to: {output_dir}")
            return self.best_code, self.best_score, output_dir
        else:
            print("\nOptimization failed: No valid solution found")
            return None, float('inf'), ""


def create_optimizer(
    op_name: str,
    op_category: str,
    tiling_resource_base: Optional[str] = None,
) -> KernelOptimizer:
    """创建 kernel 优化器"""
    return KernelOptimizer(
        op_name=op_name,
        op_category=op_category,
        tiling_resource_base=tiling_resource_base,
    )
