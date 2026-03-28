import torch
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue
import time

from config import num_correct_trials, project_root_path, seed_num

def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)

    
def execute_template(synchronize, device, context):
    correctness = True
    correctness_information = ''
    get_inputs = context['get_inputs']
    get_init_inputs = context['get_init_inputs']
    Model = context['Model']
    ModelNew = context['ModelNew']
    try:
        init_inputs = get_init_inputs()
        init_inputs = [
            x.to(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            print("using device: ", device)
            original_model = Model(*init_inputs).to(device)
            synchronize(device=device)
            set_seed(seed_num)
            custom_model = ModelNew(*init_inputs).to(device)
            synchronize(device=device)
        with torch.no_grad():
            for trial in range(num_correct_trials):
                inputs = get_inputs()
                inputs = [
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]  
                synchronize(device=device)

                new_outputs = custom_model(*inputs)
                synchronize(device=device)  # 立即同步，确保操作完成

                try:
                    ref_outputs = original_model(*inputs)
                    synchronize(device=device)  # 再次同步
                except Exception as e:
                    original_model = original_model.to('cpu')
                    _inputs = [x.to('cpu') if isinstance(x, torch.Tensor) else x for x in inputs]
                    ref_outputs = original_model(*_inputs)

                if isinstance(new_outputs, tuple) or isinstance(new_outputs, list):
                    for ref_output, new_output in zip(ref_outputs, new_outputs):
                        ref_output = ref_output.cpu().numpy()
                        new_output = new_output.cpu().numpy()

                        synchronize(device=device) # ensure all GPU operations are completed before checking results
                        feedback = None
                        if ref_output.shape != new_output.shape:
                            feedback = f"[FAIL] Output shape mismatch: Expected {ref_output.shape}, got {new_output.shape}"
                            break
                        elif not np.allclose(ref_output, new_output, atol=1e-02, rtol=1e-02):
                            # TODO: 找到ref_output和new_output中最大的差值，并打印出来
                            max_diff = np.max(np.abs(ref_output - new_output))
                            # TODO：找到ref_output和new_output中最大的差值的多维坐标索引
                            diff_index = np.unravel_index(np.argmax(np.abs(ref_output - new_output)), ref_output.shape)
                            # print(f"Max diff: {max_diff}")
                            # print("ref_output: ", str(ref_output)[:200] + "\n...\n" + str(ref_output)[-200:])
                            # print("new_output: ", str(new_output)[:200] + "\n...\n" + str(new_output)[-200:])
                            feedback = f"[FAIL] Output mismatch, max diff: {max_diff}, location: {diff_index}"
                            # TODO：打印有多少个元素不一致，及不一致的元素的多维坐标索引
                            diff_count = np.sum(np.abs(ref_output - new_output) > 1e-02)
                            diff_indices = list(zip(*np.nonzero(np.abs(ref_output - new_output) > 1e-02)))  
                            feedback += f"\nDiff count: {diff_count}"
                            feedback += f"\nDiff indices: {diff_indices[:5]}...{diff_indices[-5:]}"
                            feedback += f"\nref_output: {str(ref_output)[:200]}" + "\n...\n" + f"{str(ref_output)[-200:]}"
                            feedback += f"\nnew_output: {str(new_output)[:200]}" + "\n...\n" + f"{str(new_output)[-200:]}"
                            break
                else:
                    ref_output = ref_outputs.cpu().numpy()
                    new_output = new_outputs.cpu().numpy()

                    synchronize(device=device) # ensure all GPU operations are completed before checking results
                    feedback = None
                    if ref_output.shape != new_output.shape:
                        feedback = f"[FAIL] Output shape mismatch: Expected {ref_output.shape}, got {new_output.shape}"
                    elif not np.allclose(ref_output, new_output, atol=1e-02, rtol=1e-02):
                        # TODO: 找到ref_output和new_output中最大的差值，并打印出来
                        max_diff = np.max(np.abs(ref_output - new_output))
                        # TODO：找到ref_output和new_output中最大的差值的多维坐标索引
                        diff_index = np.unravel_index(np.argmax(np.abs(ref_output - new_output)), ref_output.shape)
                        # print(f"Max diff: {max_diff}")
                        # print("ref_output: ", str(ref_output)[:200] + "\n...\n" + str(ref_output)[-200:])
                        # print("new_output: ", str(new_output)[:200] + "\n...\n" + str(new_output)[-200:])
                        feedback = f"[FAIL] Output mismatch, max diff: {max_diff}, location: {diff_index}"
                        # TODO：打印有多少个元素不一致，及不一致的元素的多维坐标索引
                        diff_count = np.sum(np.abs(ref_output - new_output) > 1e-02)
                        diff_indices = list(zip(*np.nonzero(np.abs(ref_output - new_output) > 1e-02)))  
                        feedback += f"\nDiff count: {diff_count}"
                        feedback += f"\nDiff indices: {diff_indices[:5]}...{diff_indices[-5:]}"
                        feedback += f"\nref_output: {str(ref_output)[:200]}" + "\n...\n" + f"{str(ref_output)[-200:]}"
                        feedback += f"\nnew_output: {str(new_output)[:200]}" + "\n...\n" + f"{str(new_output)[-200:]}"

                if feedback is not None:
                    correctness = False
                    correctness_information = feedback
                    break
    except Exception as e:
        print('[FAIL] runtime error when evaluating correctness')
        correctness = False
        correctness_information = f"[FAIL] {str(e)}"
        return correctness, correctness_information

    return correctness, correctness_information
