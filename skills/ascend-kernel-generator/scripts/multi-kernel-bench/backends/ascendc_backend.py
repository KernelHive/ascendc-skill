# import torch_npu  # Defer import to avoid initializing NPU in main process
import torch
import subprocess
import shutil
from backends.backend_registry import register_backend, Backend
from utils.ascend_compile_pipeline import ascend_compile
from utils.correctness import execute_template
from utils.performance import time_execution_event_template, time_execution_event_template_with_baseline
from config import project_root_path,ascendc_device
import os
import copy
import time

@register_backend('ascendc')
class AscendBackend(Backend):
    def __init__(self):
        self.context = {}
        self._device = None
        
    def get_device(self):
        if self._device is None:
            import torch_npu
            self._device = torch.device('npu:0')
        return self._device

    def get_hardware_name(self):
        return ascendc_device

    def compile(self, generated_code, op, ref_src=""):
        try:
            # Compilation happens in a separate process, so importing here is fine if needed,
            # but ascend_compile mainly does file ops and gcc/cmake.
            # Note: ascend_compile executes model_src which imports torch_npu.
            info = ascend_compile(generated_code, op, self.context)
            return True, "", info
        except Exception as e:
            # print(f"[DEBUG] Compile failed: {str(e)}")
            os.chdir(project_root_path)
            return False, str(e), None

    def correctness_execution(self, ref_src, context=None, device=None):
        import torch_npu
        synchronize = torch_npu.npu.synchronize
            
        # Ensure device is initialized
        if self._device is None:
             self.get_device()
             
        try:
            exec(ref_src, context if context is not None else self.context)
        except Exception as e:
            raise RuntimeError(f"Failed to compile reference model: {str(e)}")
        return execute_template(synchronize, self.device if device is None else device, context if context is not None else self.context)
    
    @property
    def device(self):
        return self.get_device()

    def time_execution(self, eval_target='ModelNew', context=None, device=None):
        import torch_npu
        event_class = torch_npu.npu.Event
        synchronize = torch_npu.npu.synchronize
            
        # Ensure device is initialized
        if self._device is None:
             self.get_device()
             
        # Return both generated perf and torch_npu baseline perf (best-effort)
        return time_execution_event_template_with_baseline(
            context if context is not None else self.context,
            self.device if device is None else device,
            synchronize,
            event_class,
            eval_target=eval_target,
            baseline_target="Model",
        )

    def cleanup(self, context=None, info=None):
        # 注意：main process 不应调用 torch_npu 的操作，以防驱动挂死
        # 这里只进行文件和模块的清理
        
        # Get operator identifier from context
        context = context if context is not None else self.context
        op_identifier = info.get('op_identifier') if info else None
        
        # 清理Python模块缓存，确保模块完全卸载
        if op_identifier:
            import sys
            import importlib
            # 从sys.modules中移除相关模块（包括所有可能的变体）
            modules_to_remove = []
            for name in list(sys.modules.keys()):
                if op_identifier in name or 'custom_ops' in name.lower():
                    modules_to_remove.append(name)
            
            for module_name in modules_to_remove:
                try:
                    # 尝试清理模块的全局状态
                    module = sys.modules.get(module_name)
                    if module:
                        # 清理模块的字典
                        if hasattr(module, '__dict__'):
                            for key in list(module.__dict__.keys()):
                                if not key.startswith('__'):
                                    try:
                                        delattr(module, key)
                                    except:
                                        pass
                    del sys.modules[module_name]
                except Exception as e:
                    print(f"[WARNING] Failed to remove module {module_name}: {e}")
            
            # Uninstall the Python package
            try:
                subprocess.run(['pip', 'uninstall', op_identifier, '-y'], 
                             capture_output=True, text=True, timeout=30)
                print(f"[INFO] Uninstalled package: {op_identifier}")
            except Exception as e:
                print(f"[WARNING] Failed to uninstall package {op_identifier}: {e}")
            
            # Remove the temporary directory
            tmp_dir = os.path.join(project_root_path, 'tmp')
            op_dir = os.path.join(tmp_dir, op_identifier)
            if os.path.exists(op_dir):
                try:
                    shutil.rmtree(op_dir)
                    print(f"[INFO] Removed directory: {op_dir}")
                except Exception as e:
                    print(f"[WARNING] Failed to remove directory {op_dir}: {e}")
        
        # 清理环境变量，避免影响后续kernel
        # 清理 ASCEND_CUSTOM_OPP_PATH
        os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
        
        # 清理 LD_LIBRARY_PATH 中与当前操作符相关的路径
        if info and 'lib_path' in info:
            lib_path = info['lib_path']
            if 'LD_LIBRARY_PATH' in os.environ:
                ld_path = os.environ['LD_LIBRARY_PATH']
                # 移除包含当前操作符路径的部分
                paths = ld_path.split(':')
                # 提取要移除的路径（lib_path 的第一个路径）
                path_to_remove = lib_path.split(':')[0] if ':' in lib_path else lib_path
                filtered_paths = [p for p in paths if path_to_remove not in p]
                if filtered_paths:
                    os.environ['LD_LIBRARY_PATH'] = ':'.join(filtered_paths)
                else:
                    # 如果移除后为空，保留原始值或删除
                    original_ld_path = os.environ.get('LD_LIBRARY_PATH_ORIGINAL', '')
                    if original_ld_path:
                        os.environ['LD_LIBRARY_PATH'] = original_ld_path
                    else:
                        os.environ.pop('LD_LIBRARY_PATH', None)
        
        # 安全地清理context中的ModelNew等对象
        if context:
            keys_to_remove = ['ModelNew', 'custom_ops_lib', op_identifier] if op_identifier else ['ModelNew', 'custom_ops_lib']
            for key in keys_to_remove:
                if key in context:
                    try:
                        del context[key]
                    except:
                        pass
        
        # 增加延迟
        time.sleep(10)
