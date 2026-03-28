import numpy as np
import sys, os
import importlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_ref_src_path
from backends.backend_registry import BACKEND_REGISTRY
from utils.evaluation_utils import parser_compile_info
class Kernel:
    def __init__(self, op, category, language, ref_src_path=None, index=0, golden_src_path=None, io_desc=None):
        self.index = index
        self.op = op
        self.custom_op_name = f'{op}_custom'
        self.category = category
        self.language = language
        self.ref_src_path = ref_src_path
        ref_src_path = get_ref_src_path(op, category) if ref_src_path is None else ref_src_path
        with open(ref_src_path, 'r') as f:
            ref_src = f.read()
        self.ref_src = ref_src
        self.backend = self._init_backend(language)
        self.hardware = self.backend.get_hardware_name()
        self.context = {}
        self.golden_src_path = golden_src_path
        if golden_src_path:
            with open(golden_src_path, 'r') as f:
                self.golden_src = f.read()
        else:
            self.golden_src = None
        self.io_desc = io_desc

    def _init_backend(self, language):
        if language not in BACKEND_REGISTRY:
            try:
                importlib.import_module(f"backends.{language}_backend")
            except ImportError as e:
                raise ValueError(f"Unsupported language/platform: {language} (module not found)") from e
        backend = BACKEND_REGISTRY.get(language)
        if backend is None:
            raise ValueError(f"Unsupported language/platform: {language}")
        return backend

    def __str__(self):
        return f"{self.index} - {self.op} - {self.category}"

    def __repr__(self):
        return self.__str__()

    def set_context(self, info):
        self.context = {}
        compile(self.ref_src, "<string>", "exec")
        exec(self.ref_src, self.context)
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = info['custom_opp_path']
        os.environ["LD_LIBRARY_PATH"] = info['lib_path']
        if info.get("opp_path"):
            os.environ["ASCEND_OPP_PATH"] = info["opp_path"]
        compile(info['model_src'], "<string>", "exec")
        exec(info['model_src'], self.context)

    def compile(self, code):
        """
        Compile the generated code using the backend
        Returns: (compiled: bool, compile_info: str)
        """
        if self.backend is None:
            raise ValueError("Backend not set. Please set backend before compiling.")
        
        compiled, compile_info, info = self.backend.compile(code, self.op, self.ref_src)
        compile_info = parser_compile_info(compile_info)
        result = {'index': self.index, 'op': self.op, 'category': self.category, 'language': self.language, 'compiled': compiled, 'correctness': None, 'performance': None, 'hardware':self.hardware}
        result['compiled'] = compiled
        result['compile_info'] = compile_info
        result['info'] = info
        return result

    def cleanup(self, context=None, info=None):
        """
        Clean up backend resources
        """
        if self.backend is not None:
            try:
                self.backend.cleanup(context=context, info=info)
            except Exception as e:
                print(f"[WARNING] Failed to cleanup backend: {e}")
                pass

    def _execute_internal(self, info=None, device=None):
        """
        Internal execution logic for correctness and performance tests
        """
        # 注意：在独立的worker进程中，我们不需要重新加载模块或设置环境变量
        # 因为进程是新的，且 set_context 已经在 import 之前设置了环境变量
        
        if self.backend is None:
            raise ValueError("Backend not set. Please set backend before executing.")
        
        result = {}
        
        # 如果context为空，使用backend的context
        context_to_use = self.context if self.context else self.backend.context
        
        try:
            print(f"[DEBUG] Worker {os.getpid()} Execute correctness test")
            # Execute correctness test
            correctness, correctness_info = self.backend.correctness_execution(self.ref_src, context_to_use, device=device)
            print(f"[DEBUG] Worker {os.getpid()} Correctness test result: {correctness}")
            
            result['correctness'] = correctness
            if not correctness:
                result['correctness_info'] = correctness_info
                return result
            
            # Execute performance test. Some generated models may validate
            # correctness successfully but fail during NPU event timing.
            # Preserve correctness in that case and surface timing failure
            # through the performance payload instead of treating the whole
            # verification as incorrect.
            try:
                perf_ret = self.backend.time_execution(context=context_to_use, device=device)
            except Exception as e:
                result["performance"] = {
                    "mean": None,
                    "median": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "num_trials": 0,
                    "error": str(e),
                }
                return result

            # Backends may return:
            # - list[float]: generated-only
            # - (list[float], list[float] | None): (generated, baseline)
            if (
                isinstance(perf_ret, tuple)
                and len(perf_ret) == 2
                and isinstance(perf_ret[0], (list, tuple))
            ):
                generated_elapsed_times = list(perf_ret[0])
                baseline_elapsed_times = None if perf_ret[1] is None else list(perf_ret[1])
            else:
                generated_elapsed_times = list(perf_ret) if isinstance(perf_ret, (list, tuple)) else [perf_ret]
                baseline_elapsed_times = None

            def _summarize(times):
                if not times:
                    return {
                        "mean": None,
                        "median": None,
                        "std": None,
                        "min": None,
                        "max": None,
                        "num_trials": 0,
                    }
                arr = np.asarray(times, dtype=float)
                return {
                    "mean": float(f"{np.mean(arr):.3g}"),
                    "median": float(f"{np.median(arr):.3g}"),
                    "std": float(f"{np.std(arr):.3g}"),
                    "min": float(f"{np.min(arr):.3g}"),
                    "max": float(f"{np.max(arr):.3g}"),
                    "num_trials": int(arr.size),
                }

            # 汇总生成 kernel 的性能
            gen_stats = _summarize(generated_elapsed_times)
            result["performance"] = gen_stats

            # 如果有 baseline，则汇总 baseline，并额外计算 speedup
            if baseline_elapsed_times is not None:
                base_stats = _summarize(baseline_elapsed_times)
                result["baseline_performance"] = base_stats

                # speedup 定义为 baseline median / generated median，与 evaluation_utils 中保持一致
                try:
                    gen_med = gen_stats.get("median")
                    base_med = base_stats.get("median")
                    speedup = None
                    if gen_med not in (None, 0) and base_med is not None:
                        speedup = float(f"{base_med / gen_med:.3g}")
                    result["speedup"] = speedup
                except Exception:
                    # 不因为 speedup 计算失败而影响整体结果
                    result["speedup"] = None
        finally:
            # 执行后清理
            # 如果info不为空，说明是自定义算子，不需要在finally中清理
            # 因为我们在env.py中已经处理了进程退出
            # 但如果是单进程测试，这里可能需要清理
            if info:
                try:
                    import torch_npu
                    torch_npu.npu.empty_cache()
                except:
                    pass
        
        return result

    def execute(self, info=None, timeout=300, device=None):
        """
        Execute correctness and performance tests with timeout
        Args:
            info: Optional execution info
            timeout: Timeout in seconds (default: 300 seconds / 5 minutes)
        Returns: dict with correctness and performance results
        """
        try:
            result = self._execute_internal(info, device=device)
            return result
        except Exception as e:
            result = {
                'correctness': False,
                'correctness_info': f'Execution error: {str(e)}'
            }
            return result
        # try:
        #     with ThreadPoolExecutor(max_workers=1) as executor:
        #         future = executor.submit(self._execute_internal, info)
        #         result = future.result(timeout=timeout)
        #         return result
        # except FutureTimeoutError:
        #     result = {
        #         'correctness': False,
        #         'correctness_info': f'Execution timeout after {timeout} seconds (5 minutes)'
        #     }
        #     return result
        # except Exception as e:
        #     result = {
        #         'correctness': False,
        #         'correctness_info': f'Execution error: {str(e)}'
        #     }
        #     return result

if __name__ == "__main__":
    kernel = Kernel("mse_loss", "loss", "ascendc")
    code_path = "/home/ma-user/work/MultiKernelBench/prompts/ascendc_new_model_mse_loss.py"
    with open(code_path, 'r') as f:
        code = f.read()
    print(kernel)
    result = kernel.compile(code)
    if result['compiled']:
        kernel.set_context(result['info'])
        result.update(kernel.execute())
        print(result)
    kernel.cleanup(context=kernel.context, info=result['info'])
