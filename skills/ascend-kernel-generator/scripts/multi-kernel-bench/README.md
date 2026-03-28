> This is the parallel version of [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench)

# 环境验证

尝试跑下面的命令，如果正确输出了，则说明是环境是正常的。
```
python eval_single_runner.py /home/ma-user/work/MultiKernelBench/prompts/ascendc_new_model_mse_loss.py mse_loss ascendc result.json loss
```
输出结果：
```
[INFO] Operator project already exists, deleted
[INFO] Begin create operator project
[INFO] Create operator project succeeded
[INFO] Begin build
[INFO] Build succeeded
[INFO] Begin deploy
[INFO] Deploy succeeded
[INFO] Begin pybind
[INFO] Pybind succeeded

{'compiled': True, 'correctness': True, 'performance': {'mean': 0.181, 'std': 0.0106, 'min': 0.138, 'max': 0.215, 'num_trials': 100}, 'hardware': 'ai_core-Ascend910B4'}
```
# 并行编译 串行测试
```
python envs/env.py
```
输出结果:
```
Phase 1: Parallel compilation...
[INFO] Operator project already exists, deleted
[INFO] Begin create operator project
[INFO] Create operator project succeeded
[INFO] Begin build
[INFO] Operator project already exists, deleted
[INFO] Begin create operator project
[INFO] Create operator project succeeded
[INFO] Begin build
[INFO] Operator project already exists, deleted
......

Compile success: 3/3
Phase 2: Serial testing with process isolation...
Testing kernel 1/3
[INFO] Uninstalled package: mse_loss_custom_wqhijipbmpobxgjw
[INFO] Removed directory: /home/ma-user/work/MultiKernelBench/tmp/mse_loss_custom_wqhijipbmpobxgjw
[INFO] Uninstalled package: mse_loss_custom_wqhijipbmpobxgjw
Testing kernel 2/3
[INFO] Uninstalled package: mse_loss_custom_udrfcrzhpatputsr
[INFO] Removed directory: /home/ma-user/work/MultiKernelBench/tmp/mse_loss_custom_udrfcrzhpatputsr
[INFO] Uninstalled package: mse_loss_custom_udrfcrzhpatputsr
Testing kernel 3/3
[INFO] Uninstalled package: mse_loss_custom_qzshiixeqnkspxxp
[INFO] Removed directory: /home/ma-user/work/MultiKernelBench/tmp/mse_loss_custom_qzshiixeqnkspxxp
[INFO] Uninstalled package: mse_loss_custom_qzshiixeqnkspxxp
[{'index': 0, 'op': 'mse_loss', 'category': 'loss', 'language': 'ascendc', 'compiled': True, 'correctness': True, 'performance': {'mean': 0.194, 'std': 0.0334, 'min': 0.157, 'max': 0.521, 'num_trials': 100}, 'hardware': 'ai_core-Ascend910B4', 'compile_info': None}, {'index': 1, 'op': 'mse_loss', 'category': 'loss', 'language': 'ascendc', 'compiled': True, 'correctness': True, 'performance': {'mean': 0.186, 'std': 0.00888, 'min': 0.139, 'max': 0.227, 'num_trials': 100}, 'hardware': 'ai_core-Ascend910B4', 'compile_info': None}, {'index': 2, 'op': 'mse_loss', 'category': 'loss', 'language': 'ascendc', 'compiled': True, 'correctness': True, 'performance': {'mean': 0.183, 'std': 0.00896, 'min': 0.139, 'max': 0.198, 'num_trials': 100}, 'hardware': 'ai_core-Ascend910B4', 'compile_info': None}]
```