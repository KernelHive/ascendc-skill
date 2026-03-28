from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool
import time
import subprocess
import json
import os
import multiprocessing
import sys
import threading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kernel import Kernel

# Set multiprocessing start method to 'spawn' for NPU compatibility
multiprocessing.set_start_method('spawn', force=True)

# 全局TBE错误过滤器类
class TBEErrorFilter:
    """过滤TBE内部线程的预期错误"""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.tbe_error_patterns = [
            'EOFError',
            'ConnectionResetError',
            'Connection reset by peer',
            'ForkAwareLocal',
            'multiprocess_util.py',
            'multiprocessing/managers.py',
            'multiprocessing/connection.py',
            'multiprocessing/util.py',
            'Exception in thread Thread-1',
            'During handling of the above exception',
            'tbe/common/repository_manager',
            'cann_kb_manager',
            'knowledge_bank_manager',
            'base_manager.py',
            'route.py',
            'answer_challenge',
            'recv_bytes',
            '_recv_bytes',
            '_recv(',
            'Client(',
            '_connect()',
            'AttributeError: \'ForkAwareLocal\'',
            'raise EOFError',
        ]
        self.tbe_keywords = [
            'tbe',
            'repository_manager',
            'multiprocess_util',
            'task_distribute',
            'resource_tracker',
            'spawn_main'
        ]
        self.buffer = ''
        self.in_tbe_traceback = False
        self.traceback_lines = []
    
    def write(self, text):
        # 累积多行错误信息
        self.buffer += text
        self.traceback_lines.append(text)

    def flush(self):
        if self.buffer:
            # 检查缓冲区中剩余的内容
            buffer_lower = self.buffer.lower()
            is_tbe_error = (
                self.in_tbe_traceback or
                any(pattern in self.buffer for pattern in self.tbe_error_patterns) or
                any(keyword in buffer_lower for keyword in self.tbe_keywords)
            )
            if not is_tbe_error and self.buffer.strip():
                self.original_stderr.write(self.buffer)
            self.buffer = ''
            self.traceback_lines = []
            self.in_tbe_traceback = False
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        # 代理其他属性到原始stderr
        return getattr(self.original_stderr, name)

# 设置全局stderr过滤（在主进程中）
_original_stderr = sys.stderr
_global_tbe_filter = None

def _setup_global_stderr_filter():
    """在主进程中设置全局stderr过滤"""
    global _global_tbe_filter, _original_stderr
    if _global_tbe_filter is None:
        _global_tbe_filter = TBEErrorFilter(_original_stderr)
        sys.stderr = _global_tbe_filter

# 初始化全局过滤器
_setup_global_stderr_filter()

def _cleanup_tbe_processes():
    """
    清理可能的TBE残留进程
    策略:
    1. 清理当前进程树下的TBE进程
    2. 清理当前用户下的孤儿TBE进程(PPID=1)
    3. 保留其他非孤儿TBE进程(可能属于并行运行的其他任务)
    """
    try:
        import psutil
        current_pid = os.getpid()
        current_proc = psutil.Process(current_pid)
        try:
            current_user = current_proc.username()
        except Exception:
            current_user = None

        tbe_processes = []
        
        # 获取当前进程的所有后代PID集合，用于快速判断
        try:
            my_descendants = {p.pid for p in current_proc.children(recursive=True)}
        except Exception:
            my_descendants = set()

        # 查找所有进程
        for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline', 'username']):
            try:
                # Only clean processes of current user
                if current_user and proc.info['username'] != current_user:
                    continue
                
                # Skip current process
                if proc.pid == current_pid:
                    continue
                
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                cmdline_lower = cmdline.lower()
                
                # Check if it's a TBE related process
                is_tbe = ('task_distribute' in cmdline_lower or 
                          'tbe' in cmdline_lower or 
                          'te.platform' in cmdline_lower or
                          'resource_tracker' in cmdline_lower or
                          'spawn_main' in cmdline_lower)
                
                if not is_tbe:
                    continue

                # 判定是否需要清理
                # 1. 是当前进程的后代
                if proc.pid in my_descendants or proc.info['ppid'] == current_pid:
                    tbe_processes.append(proc)
                    continue
                
                # 2. 是孤儿进程 (PPID=1)，认为是之前运行残留的垃圾
                # 注意: 如果系统中有正常的长期运行的TBE服务(非一次性)，这可能会误杀。
                # 但在Benchmark环境中，通常都是一次性任务。
                if proc.info['ppid'] == 1:
                    tbe_processes.append(proc)
                    continue
                
                # 3. 其他情况(PPID!=1 且不是我的后代)，认为是并行运行的其他任务，跳过
                
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # 先尝试优雅终止
        for proc in tbe_processes:
            try:
                if proc.is_running():
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # 等待进程退出
        if tbe_processes:
            try:
                psutil.wait_procs(tbe_processes, timeout=2)
            except:
                pass
        
        # 强制kill
        for proc in tbe_processes:
            try:
                if proc.is_running():
                    # print(f"[INFO] Force killing TBE process: {proc.pid}")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
                
    except ImportError:
        # psutil不可用，降级使用pkill（注意：pkill无法区分PPID，可能会误杀并行任务，需谨慎）
        # 在并行环境下最好安装psutil
        try:
            # 仅清理 orphan 比较难用简单的shell命令实现，这里只清理明显是子进程的
            # 或者不做处理，依赖系统清理
            pass
        except:
            pass
    except Exception as e:
        print(f"[WARNING] Failed to cleanup TBE processes: {e}")

def _worker_process(kernel_data, info_data, device=None):
    """
    Worker function for executing kernel in separate process
    """
    kernel = None
    try:
        print(f"[DEBUG] Worker process {os.getpid()} started on device {device}")
        
        # 设置环境变量 (关键：在导入任何后端库之前)
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = info_data['custom_opp_path']
        os.environ["LD_LIBRARY_PATH"] = info_data['lib_path']
        if info_data.get("opp_path"):
            os.environ["ASCEND_OPP_PATH"] = info_data["opp_path"]
        
        # 设置设备ID环境变量
        if device is not None:
            if isinstance(device, str) and ':' in device:
                device_id = device.split(':')[1]
            else:
                device_id = str(device)
            os.environ["ASCEND_DEVICE_ID"] = device_id
            print(f"[DEBUG] Worker {os.getpid()} set ASCEND_DEVICE_ID={device_id}")
        
        print(
            f"[DEBUG] Worker {os.getpid()} Env vars set: "
            f"CUSTOM_OPP={info_data['custom_opp_path']}, "
            f"OPP={info_data.get('opp_path')}"
        )
        
        # 在worker进程中清理可能的残留资源
        try:
            import torch_npu
            print(f"[DEBUG] Worker {os.getpid()} imported torch_npu")
            
            # 设置torch设备
            if device is not None:
                if isinstance(device, str) and ':' in device:
                    device_id = int(device.split(':')[1])
                else:
                    device_id = int(device) if not isinstance(device, int) else device
                torch_npu.npu.set_device(device_id)
                print(f"[DEBUG] Worker {os.getpid()} set torch device to npu:{device_id}")
            
            # 多次清理，确保资源完全释放
            for _ in range(3):
                torch_npu.npu.empty_cache()
                try:
                    torch_npu.npu.synchronize()
                except:
                    pass
            print(f"[DEBUG] Worker {os.getpid()} NPU cache cleared")
        except:
            print(f"[DEBUG] Worker {os.getpid()} NPU cleanup failed/skipped")
            pass
        
        kernel = Kernel(
            kernel_data['op'],
            kernel_data['category'],
            kernel_data['language'],
            kernel_data['ref_src_path'],
            index=kernel_data['index'],
        )
        print(f"[DEBUG] Worker {os.getpid()} Kernel object created")
        kernel.set_context(info_data)
        print(f"[DEBUG] Worker {os.getpid()} Kernel context set")
        
        print(f"[DEBUG] Worker {os.getpid()} executing kernel...")
        result = kernel.execute(info_data, device=device)
        print(f"[DEBUG] Worker {os.getpid()} kernel execution finished")
        
        # 执行后立即清理
        if kernel:
            kernel.cleanup(context=kernel.context, info=info_data)
        
        # 再次清理NPU资源（多次清理，特别是对于使用workspace的操作符）
        try:
            import torch_npu
            for _ in range(3):
                torch_npu.npu.empty_cache()
                try:
                    torch_npu.npu.synchronize()
                except:
                    pass
        except:
            pass
        
        return result
    except Exception as e:
        # 确保即使出错也清理资源
        if kernel:
            try:
                kernel.cleanup(context=kernel.context if hasattr(kernel, 'context') else {}, info=info_data)
            except:
                pass
        # 出错时也要清理NPU资源
        try:
            import torch_npu
            for _ in range(3):
                torch_npu.npu.empty_cache()
                try:
                    torch_npu.npu.synchronize()
                except:
                    pass
        except:
            pass
        raise


def _parallel_test_worker(i, kernel_data, result_dict, info_data, device, total_kernels):
    """
    顶级函数：在独立进程中运行单个测试任务
    用于ProcessPoolExecutor的并行测试
    """
    try:
        if not result_dict['compiled']:
            print(f"[Process {os.getpid()}] Kernel {i+1} not compiled")
            if 'info' in result_dict:
                del result_dict['info']
            return i, result_dict

        print(f"[Process {os.getpid()}] Testing kernel {i+1}/{total_kernels} on device {device}")
        # 设置设备环境变量（在导入torch_npu之前）
        if isinstance(device, str) and ':' in device:
            device_id = device.split(':')[1]
        else:
            device_id = str(device)
        os.environ["ASCEND_DEVICE_ID"] = device_id
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = info_data['custom_opp_path']
        os.environ["LD_LIBRARY_PATH"] = info_data['lib_path']
        if info_data.get("opp_path"):
            os.environ["ASCEND_OPP_PATH"] = info_data["opp_path"]
        
        # 重建kernel对象（在新进程中）
        kernel = Kernel(
            kernel_data['op'],
            kernel_data['category'],
            kernel_data['language'],
            kernel_data['ref_src_path'],
            index=kernel_data['index'],
        )
        
        result = result_dict.copy()
        
        try:
            # 导入torch_npu并设置设备
            import torch_npu
            device_id_int = int(device_id)
            torch_npu.npu.set_device(device_id_int)
            print(f"[Process {os.getpid()}] Set torch device to npu:{device_id_int}")
            
            # 清理缓存
            for _ in range(3):
                torch_npu.npu.empty_cache()
                try:
                    torch_npu.npu.synchronize()
                except:
                    pass
            
            # 设置context
            kernel.set_context(info_data)
            
            # 执行测试
            execute_result = kernel._execute_internal(info_data, device=device)
            result.update(execute_result)
            
        except Exception as e:
            import traceback
            result.update({
                'correctness': False,
                'correctness_info': f"Execution failed: {str(e)}\n{traceback.format_exc()}"
            })
        finally:
            try:
                kernel.cleanup(context=kernel.context if hasattr(kernel, 'context') else {}, info=info_data)
            except:
                pass
            
            # 清理NPU资源
            try:
                import torch_npu
                for _ in range(3):
                    torch_npu.npu.empty_cache()
                    try:
                        torch_npu.npu.synchronize()
                    except:
                        pass
            except:
                pass
        
        if 'info' in result:
            del result['info']
        
        return i, result
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Worker process error: {e}")
        traceback.print_exc()
        return i, {
            'correctness': False,
            'correctness_info': f'Worker process error: {str(e)}'
        }


def _worker_process_entry(queue, kernel_data, info_data, device=None):
    """
    Entry point for subprocess execution that communicates results via a queue.
    """
    # 重定向stderr来抑制TBE内部线程的错误输出
    import sys
    
    # 在子进程中也设置stderr过滤
    original_stderr = sys.stderr
    filtered_stderr = TBEErrorFilter(original_stderr)
    sys.stderr = filtered_stderr
    
    # 创建新的进程组，方便清理所有子进程
    try:
        os.setsid()  # 创建新的会话和进程组
    except:
        pass  # 如果已经是在新进程中，可能会失败，忽略
    
    # 设置信号处理，确保收到终止信号时能优雅退出
    import signal
    import atexit
    
    def cleanup_and_exit():
        """清理资源"""
        # 注意：在信号处理或退出处理中，尽量避免调用可能阻塞的NPU操作(如synchronize)
        
        # 清理TBE子进程
        try:
            import psutil
            current = psutil.Process()
            children = current.children(recursive=True)
            for child in children:
                try:
                    if child.is_running():
                        child.terminate()
                except:
                    pass
            if children:
                try:
                    psutil.wait_procs(children, timeout=1)
                except:
                    pass
        except:
            pass
        # 恢复原始stderr
        try:
            sys.stderr = original_stderr
        except:
            pass
            
    def force_exit_handler(signum, frame):
        """信号处理函数"""
        cleanup_and_exit()
        os._exit(1) # 信号处理需要强制退出
    
    # 注册退出时的清理函数 (正常退出时调用)
    atexit.register(cleanup_and_exit)
    
    # 设置信号处理
    signal.signal(signal.SIGTERM, force_exit_handler)
    signal.signal(signal.SIGINT, force_exit_handler)
    
    try:
        result = _worker_process(kernel_data, info_data, device=device)
        # queue now is a connection object (Pipe)
        queue.send(('result', result))
        queue.close()
    except KeyboardInterrupt:
        force_exit_handler(signal.SIGINT, None)
    except (EOFError, ConnectionResetError, BrokenPipeError) as e:
        force_exit_handler(signal.SIGTERM, None)
    except Exception as exc:
        import traceback
        try:
            queue.send((
                'error',
                {
                    'type': exc.__class__.__name__,
                    'message': str(exc),
                    'traceback': traceback.format_exc(),
                },
            ))
            queue.close()
        except:
            print(f"[ERROR] Worker process error: {exc}")
            traceback.print_exc()
    finally:
        # 正常退出路径，不需要 os._exit，让 Python 运行时处理资源释放
        pass

class Env:
    def __init__(self, op, category, language, ref_src_path=None, max_workers=10, devices=None):
        _cleanup_tbe_processes()
        self.op = op
        self.category = category
        self.language = language
        self.ref_src_path = ref_src_path
        self.max_workers = max_workers
        self.devices = devices
        
    def reset(self):
        pass

    def _check_npu_health(self):
        """
        Uses utils/check_npu.py to check NPU status.
        Returns True if healthy, False otherwise.
        """
        try:
            # Use python from current environment
            python_executable = sys.executable
            # Resolve path to utils/check_npu.py relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            check_script = os.path.join(base_dir, 'utils', 'check_npu.py')

            # Determine device argument (optional)
            # If self.devices is not set, let check_npu.py use its default ('npu:0')
            cmd = [python_executable, check_script]
            if self.devices:
                try:
                    # Support both list (['npu:0', ...]) and single string ('npu:0')
                    first_device = self.devices[0] if isinstance(self.devices, (list, tuple)) else self.devices
                    cmd.extend(['--device', str(first_device)])
                except Exception:
                    # Fallback: no explicit device flag; rely on default in check_npu.py
                    pass

            # Run check_npu.py with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 60s timeout for check
            )
            if result.returncode == 0:
                # Check stdout for success message just in case
                if "[SUCCESS]" in result.stdout:
                    return True
            
            # Print stderr for debugging if failed
            # print(f"[DEBUG] NPU Check failed: {result.stderr}")
            return False
            
        except subprocess.TimeoutExpired:
            # print("[DEBUG] NPU Check timed out")
            return False
        except Exception as e:
            print(f"[WARNING] NPU Check exception: {e}")
            return False

    def step(self, codes):
        # Pre-flight NPU Health Check with Retry
        # Ensures we start with a clean slate, similar to test3.sh logic
        max_retries = 5
        for attempt in range(max_retries):
            if self._check_npu_health():
                if attempt > 0:
                    print(f"[INFO] NPU recovered on attempt {attempt + 1}")
                break
            
            print(f"[WARNING] NPU unhealthy (Attempt {attempt + 1}/{max_retries}). Cleaning up...")
            _cleanup_tbe_processes()
            # Sleep increasing amount
            time.sleep(5 + attempt * 5)
        else:
            print("[ERROR] NPU is persistently unhealthy. Aborting step.")
            # Return error results for all codes
            return [{
                'compiled': False,
                'correctness': False, 
                'correctness_info': 'NPU persistent failure before execution',
                'info': None
            } for _ in codes]

        _cleanup_tbe_processes()
        kernels = [Kernel(self.op, self.category, self.language, self.ref_src_path, index=i) for i in range(len(codes))]
        results = []
        
        # 阶段1：并行编译所有kernels
        print("Phase 1: Parallel compilation...")
        futures = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for i, (kernel, code) in enumerate(zip(kernels, codes)):
                futures.append(executor.submit(kernel.compile, code))
                time.sleep(2)  # 增加延迟时间
        
        # 编译阶段也设置超时（5分钟）
        _results = []
        for idx, future in enumerate(futures):
            try:
                result = future.result(timeout=300)  # 5分钟超时
                _results.append(result)
            except FutureTimeoutError:
                print(f"[WARNING] Compilation timeout after 5 minutes for kernel {idx}")
                _results.append({
                    'compiled': False,
                    'compile_info': 'Compilation timeout after 5 minutes',
                    'info': None
                })
            except BrokenProcessPool as e:
                print(f"[ERROR] Process pool broken during compilation for kernel {idx}: {e}")
                print(f"[ERROR] This usually means a worker process crashed or was killed")
                import traceback
                traceback.print_exc()
                _results.append({
                    'compiled': False,
                    'compile_info': f'Worker process crashed or was killed during compilation: {str(e)}. This may be due to device access issues, memory problems, or other system-level errors.',
                    'info': None
                })
            except Exception as e:
                print(f"[WARNING] Compilation error for kernel {idx}: {e}")
                import traceback
                traceback.print_exc()
                _results.append({
                    'compiled': False,
                    'compile_info': f'Compilation error: {str(e)}',
                    'info': None
                })
        
        compile_success = [result['compiled'] for result in _results]
        print(f"Compile success: {compile_success.count(True)}/{len(kernels)}")
        
        # 阶段2：串行测试已编译的kernels（使用进程隔离）
        print("Phase 2: Serial testing with process isolation...")
        if self.devices and len(self.devices) > 0:
            print(f"Phase 2: Parallel testing on {len(self.devices)} devices: {self.devices}")
            
            # 准备任务数据
            tasks_data = []
            for i, (kernel, _result) in enumerate(zip(kernels, _results)):
                device = self.devices[i % len(self.devices)]
                kernel_data = {
                    'op': kernel.op,
                    'category': kernel.category,
                    'language': kernel.language,
                    'ref_src_path': kernel.ref_src_path,
                    'index': kernel.index
                }
                tasks_data.append((i, kernel_data, _result, _result['info'], device, len(kernels)))
            
            # 使用ProcessPoolExecutor进行真正的多进程并行
            parallel_results = []
            with ProcessPoolExecutor(max_workers=len(self.devices)) as executor:
                futures = []
                for task_data in tasks_data:
                    future = executor.submit(_parallel_test_worker, *task_data)
                    futures.append(future)
                
                # 收集结果
                timeout_seconds = 600
                for idx, future in enumerate(futures):
                    try:
                        result = future.result(timeout=timeout_seconds)  # 给每个任务6分钟超时
                        parallel_results.append(result)
                    except FutureTimeoutError:
                        print(f"[WARNING] Task {idx} timeout after {timeout_seconds} seconds")
                        parallel_results.append((idx, {
                            'correctness': False,
                            'correctness_info': f'Task timeout after {timeout_seconds} seconds'
                        }))
                    except BrokenProcessPool as e:
                        print(f"[ERROR] Process pool broken for task {idx}: {e}")
                        print(f"[ERROR] This usually means a worker process crashed or was killed")
                        import traceback
                        traceback.print_exc()
                        # 尝试从 tasks_data 获取原始索引
                        task_idx = tasks_data[idx][0] if idx < len(tasks_data) else idx
                        parallel_results.append((task_idx, {
                            'correctness': False,
                            'correctness_info': f'Worker process crashed or was killed: {str(e)}. This may be due to device access issues, memory problems, or other system-level errors.'
                        }))
                    except Exception as e:
                        import traceback
                        print(f"[WARNING] Task {idx} execution error: {e}")
                        traceback.print_exc()
                        # 尝试从 tasks_data 获取原始索引
                        task_idx = tasks_data[idx][0] if idx < len(tasks_data) else idx
                        parallel_results.append((task_idx, {
                            'correctness': False,
                            'correctness_info': f'Task execution error: {str(e)}'
                        }))
            
            # Sort results by index to maintain order
            parallel_results.sort(key=lambda x: x[0])
            results = [r[1] for r in parallel_results]
            
            # 清理kernels（在主进程中）
            for  i, (kernel, _result) in enumerate(zip(kernels, _results)):
                try:
                    kernel.cleanup(context=kernel.context, info=_result['info'])
                except:
                    pass

        else:
            # Serial testing for when no specific devices are provided
            for i, (kernel, _result) in enumerate(zip(kernels, _results)):
                result = _result
                
                # 在每个kernel测试前清理资源
                # 主进程不应触碰NPU，避免因设备挂死导致主进程卡住
                
                if _result['compiled']:
                    print(f"Testing kernel {i+1}/{len(kernels)}")
                    # 使用进程隔离执行kernel
                    # 恢复正确的超时时间设置
                    op_timeout = 300
                    execute_result = self._execute_kernel_in_process(kernel, _result['info'], timeout=op_timeout)
                    result.update(execute_result)
                    # 立即清理kernel资源
                    kernel.cleanup(context=kernel.context, info=_result['info'])
                else:
                    kernel.cleanup(context=kernel.context, info=_result['info'])
                
                if 'info' in result:
                    del result['info']
                results.append(result)
                
                # 添加延迟以确保设备资源完全释放（特别是对于使用workspace的操作符）
                if i < len(kernels) - 1:  # 不是最后一个kernel
                    # 清理可能的TBE残留进程
                    _cleanup_tbe_processes()
                    
                    # 主进程不直接操作NPU，依靠子进程退出时的OS资源回收
                    time.sleep(5)  # 增加延迟时间，等待OS回收资源
                    
                    # 再次清理TBE进程
                    _cleanup_tbe_processes()
        
        return results

    def _execute_kernel_in_process(self, kernel, info, timeout=300, device=None):
        """
        在独立进程中执行kernel，避免操作符注册冲突
        """
        # Double check NPU health before starting a new worker
        # This helps if previous kernels in the loop left NPU in bad state
        # Try quick check
        if not self._check_npu_health():
             print("[WARNING] NPU unhealthy before kernel execution. Attempting cleanup...")
             _cleanup_tbe_processes()
             time.sleep(5)
             if not self._check_npu_health():
                 return {
                     'correctness': False,
                     'correctness_info': 'NPU unhealthy before execution'
                 }

        kernel_data = {
            'op': kernel.op,
            'category': kernel.category,
            'language': kernel.language,
            'ref_src_path': kernel.ref_src_path,
            'index': kernel.index
        }
        
        ctx = multiprocessing.get_context('spawn')
        # 使用Pipe代替Queue，避免SIGKILL导致的信号量泄露
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        
        # 使用start_new_session创建新的进程组，方便清理所有子进程
        process = ctx.Process(
            target=_worker_process_entry,
            args=(child_conn, kernel_data, info, device),
        )
        process.start()
        process_pid = process.pid
        
        # 父进程不需要child_conn
        child_conn.close()
        
        # 等待一小段时间，确保进程完全启动
        time.sleep(0.1)
        
        # 等待进程完成或超时
        process.join(timeout)
        
        if process.is_alive():
            # 进程超时，需要强制终止
            print(f"[WARNING] Process timeout, terminating process {process_pid}")
            # 尝试打印堆栈(如果可能) - 这里只能打印主进程的，子进程的很难获取
            
            try:
                # 1. 先尝试清理TBE子进程（防止它们卡住）
                _cleanup_tbe_processes()
                
                # 2. 尝试优雅终止主进程
                process.terminate()
                process.join(timeout=2)  # 只等2秒，不要等太久
                
                # 3. 如果还在运行，强制kill
                if process.is_alive():
                    print(f"[WARNING] Process did not exit gracefully, force killing...")
                    # 使用进程组来终止所有子进程
                    try:
                        import psutil
                        try:
                            parent = psutil.Process(process_pid)
                            children = parent.children(recursive=True)
                            for child in children:
                                try:
                                    if child.is_running():
                                        child.kill()
                                except:
                                    pass
                        except:
                            pass
                    except:
                        pass
                    
                    # 强制kill主进程
                    import signal
                    try:
                        os.killpg(process_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except Exception:
                         try:
                             os.kill(process_pid, signal.SIGKILL)
                         except:
                             pass
                    process.join(timeout=1)
                
                # 4. 再次清理所有TBE相关进程
                _cleanup_tbe_processes()
                
                # 5. 移除主进程NPU清理，避免卡死
                
                # Pipe 不需要显式清理逻辑，GC会自动关闭文件描述符
                
            except Exception as e:
                print(f"[WARNING] Error terminating process: {e}")
                _cleanup_tbe_processes()
                # 最后的保底清理
                try:
                    import signal
                    os.kill(process_pid, signal.SIGKILL)
                except:
                    pass
            
            return {
                'correctness': False,
                'correctness_info': f'Execution timeout after {timeout} seconds'
            }
        
        # 进程正常退出，清理可能的残留TBE进程
        _cleanup_tbe_processes()
        
        # 移除主进程NPU清理
        
        # 检查是否有数据
        if not parent_conn.poll():
             # 进程已退出但没有数据，可能是crash了
             pass
        else:
            try:
                status, payload = parent_conn.recv()
                if status == 'result':
                    return payload
                
                raise RuntimeError(
                    f"Worker process raised {payload['type']}: {payload['message']}\n{payload['traceback']}"
                )
            except EOFError:
                pass
        
        # 如果没有收到结果
        return {
             'correctness': False,
             'correctness_info': 'Worker process exited without returning a result (likely crashed or killed)'
        }

if __name__ == "__main__":
    env = Env(op="gelu_mul", category="activation", language="ascendc")
    code_path = "/home/ma-user/work/MultiKernelBench/test/golden_solution/activation/gelu_mul_no_queue.py"
    codes = [open(os.path.join(code_path), 'r').read() for i in range(2)]
    results = env.step(codes)
    print(results)
