import re, os, json
from utils.utils import get_ref_src_path
from backends.backend_registry import BACKEND_REGISTRY
import torch
import importlib
import numpy as np
from dataset import dataset
from config import temperature, top_p, num_perf_trials
import random, string

def parser_compile_info(compile_info: str) -> str:
    print(compile_info)
    """
    Parse compile info from model output
    """
    # breakpoint()
    error = re.findall(
        r'(op_(?:kernel|host)/.*?error:.*)',
        compile_info,
    )
    if error:
        error = list(set(error))
        error = '\n'.join(error)
        return error
    return compile_info

def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code_block = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code_block.startswith(code_type):
                code_block = code_block[len(code_type) :].strip()
        return code_block

    return None

def eval_single(response_txt:str, op, language, category=None, cache_dir=None, ref_src_path=None):
    # Try to dynamically import the backend if it's not yet registered
    if language not in BACKEND_REGISTRY:
        try:
            importlib.import_module(f"backends.{language}_backend")
        except ImportError as e:
            raise ValueError(f"Unsupported language/platform: {language} (module not found)") from e
    backend = BACKEND_REGISTRY.get(language)
    if backend is None:
        raise ValueError(f"Unsupported language/platform: {language}")
    
    hardware = backend.get_hardware_name()
    if ref_src_path is None:
        ref_src_path = get_ref_src_path(op, category)
    with open(ref_src_path, 'r') as f:
        ref_src = f.read()

    if cache_dir is None and hasattr(backend, 'set_cache_dir'):
        cache_dir = "./cache/" + op + '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        backend.set_cache_dir(cache_dir)
    
    result = {'compiled': False, 'correctness': None, 'performance': None, 'hardware':hardware}
    generated_code = extract_first_code(response_txt, ['python', 'cpp'])
    if generated_code is None:
        generated_code = response_txt

    compiled, compile_info, info = backend.compile(generated_code, op, ref_src)
    if not compiled:
        result['compile_info'] = parser_compile_info(compile_info)
        backend.cleanup()
        return result
    result['compiled'] = True
    
    correctness, info = backend.correctness_execution(ref_src)
    if not correctness:
        result['correctness_info'] = info
        backend.cleanup()
        return result
    result['correctness'] = True
    elapsed_times, baseline_elapsed_times = backend.time_execution()
    baseline_stats = {
        "baseline_mean": None,
        "baseline_median": None,
        "baseline_std": None,
        "baseline_min": None,
        "baseline_max": None,
    }
    speedup = None
    try:
        if baseline_elapsed_times is not None and len(baseline_elapsed_times) > 0:
            baseline_stats = {
                "baseline_mean": float(f"{np.mean(baseline_elapsed_times):.3g}"),
                "baseline_median": float(f"{np.median(baseline_elapsed_times):.3g}"),
                "baseline_std": float(f"{np.std(baseline_elapsed_times):.3g}"),
                "baseline_min": float(f"{np.min(baseline_elapsed_times):.3g}"),
                "baseline_max": float(f"{np.max(baseline_elapsed_times):.3g}"),
            }
            gen_median = float(f"{np.median(elapsed_times):.3g}")
            if gen_median != 0:
                speedup = float(f"{(baseline_stats['baseline_median'] / gen_median):.3g}")
    except Exception:
        baseline_stats = {
            "baseline_mean": None,
            "baseline_median": None,
            "baseline_std": None,
            "baseline_min": None,
            "baseline_max": None,
        }
        speedup = None
    result['performance'] = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "median": float(f"{np.median(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
        # Keep existing key for backward compatibility (baseline median).
        "baseline": baseline_stats["baseline_median"],
        **baseline_stats,
        "speedup": speedup,
    }
    backend.cleanup()
    return result

def eval_all(out_dir, language, op_tested=dataset.keys(), category=None):
    result = {}
    
    for op in op_tested:
        print(f"[INFO] eval op {op}")
        with open(os.path.join(out_dir, f'{op}.txt'), 'r') as saved_log:
            response_txt = saved_log.read()
        result[op] = eval_single(response_txt, op, language, category)
        
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    
if __name__ == '__main__':
    runs = 1
    model = 'deepseek-chat'
    language = 'cuda'
    op_tested = list(dataset.keys())
    op_tested = ['ltsm_hn', 'conv3d_leaky_relu_sum_clamp_gelu','square_matrix_multiplication','l2_norm','adam','sgd']
    select_shot = False
    for run in range(runs):
        if not select_shot:
            out_dir = f'output/{language}/add_shot/{temperature}-{top_p}/{model}/run{run}'
        else:
            out_dir = f'output/{language}/selected_shot/{temperature}-{top_p}/{model}/run{run}'
        eval_all(out_dir, language, op_tested)
