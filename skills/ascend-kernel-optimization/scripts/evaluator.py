"""
Kernel Evaluator for ascend-kernel-optimization skill.
负责调用评估服务获取性能分数。
"""

import os
import re
from typing import Dict, Optional, Tuple

import requests


class KernelEvaluator:
    """Kernel 优化专用的评估器"""

    def __init__(
        self,
        log_dir: str,
        op_name: str,
        op_category: str,
        tiling_resource_base: Optional[str] = None,
    ):
        self.log_dir = log_dir
        self.op_name = op_name
        self.op_category = op_category
        self.tiling_resource_base = tiling_resource_base
        
        self.service_url = os.environ.get("EVALUATE_SERVICE_URL", "http://127.0.0.1:6666/evaluate")
        self.timeout = 1800
        self.failure_score = -10000
        self.base_op_file_list = None

    def set_base_op_files(self, op_file_list: Dict[str, str]):
        """设置基础算子文件"""
        self.base_op_file_list = op_file_list.copy()

    def call_evaluate_service(
        self, 
        op_name: str, 
        op_file_list: Dict[str, str]
    ) -> float:
        """
        调用评估服务并获取分数
        
        Args:
            op_name: 算子名称
            op_file_list: 替换的文件内容
            
        Returns:
            float: 评估分数
        """
        url = self.service_url
        headers = {"Content-Type": "application/json"}
        payload = {
            "op_name": op_name,
            "op_file_list": op_file_list,
            "mode": "cann_nightly",
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "success":
                score_data = result.get("score")
                if isinstance(score_data, dict):
                    score = score_data.get("score_val", self.failure_score)
                else:
                    score = score_data if score_data is not None else self.failure_score
                print(f"[Success] Op: {op_name}, Score: {score}")
                return score
            else:
                print(f"[Error] {result.get('error_info', 'Unknown error')}")
                return self.failure_score

        except Exception as e:
            print(f"[Error] Request failed: {e}")
            return self.failure_score

    def get_op_file_list(self) -> Dict[str, str]:
        """获取算子文件列表"""
        base_path = os.path.join(self.tiling_resource_base, self.op_category, self.op_name)
        op_file_list = {}
        
        for subdir in ["op_host", "op_kernel"]:
            dir_path = os.path.join(base_path, subdir)
            if not os.path.exists(dir_path):
                continue

            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        rel_path = os.path.relpath(file_path, base_path)
                        op_file_list[rel_path] = content
                    except:
                        pass

        cmake_file_path = os.path.join(base_path, "CMakeLists.txt")
        if os.path.exists(cmake_file_path):
            with open(cmake_file_path, "r", encoding="utf-8") as f:
                op_file_list["CMakeLists.txt"] = f.read()

        return op_file_list

    def evaluate(
        self, 
        modified_codes: Dict[str, str], 
        log_dir: str, 
        island_id: int, 
        iteration: int
    ) -> Tuple[bool, float]:
        """
        评估修改后的代码
        
        Args:
            modified_codes: 修改后的文件 {"relative/path": "content"}
            log_dir: 日志目录
            island_id: 岛屿ID
            iteration: 迭代次数
            
        Returns:
            (success, score)
        """
        if not modified_codes:
            return False, self.failure_score

        if self.base_op_file_list:
            op_file_list = self.base_op_file_list.copy()
        else:
            op_file_list = self.get_op_file_list()

        # 合并修改
        for rel_path, content in modified_codes.items():
            op_file_list[rel_path] = content

        score = self.call_evaluate_service(
            f"{self.op_category}/{self.op_name}", 
            op_file_list
        )
        
        success = score > self.failure_score
        
        return success, score

