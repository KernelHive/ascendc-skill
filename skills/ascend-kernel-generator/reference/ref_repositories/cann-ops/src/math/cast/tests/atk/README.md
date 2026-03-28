# ATK测试

本节介绍如何运行和编写 `cast` 模块的测试用例。

## 运行测试

要运行测试，请确保您已经安装了必要的依赖项。然后在项目根目录下执行以下命令：

```bash
pip3 install ATK*.whl
```

## 用例设计

根据文档，设计用例生成的各类参数的yaml文件。

## 用例生成

```bash
export ATK_TASK_OUTPUT_PATH="${git_clone_path}/cann-ops/src/math/cast/tests/atk"
atk case -f op_cast.yaml -p generate_cast.py 
```

## aclnn算子验证

| 任务场景   | 测试算子  | 标杆算子    |
|------------|-----------|-------------|
| 精度比对   | aclnn     | cpu         |
| 确定性计算 | aclnn     | cpu         |
| 性能比对   | aclnn     | aclnn/npu   |

### 进行aclnnCast算子和cpu上的torch.cast算子的精度比对

```bash
atk node --backend pyaclnn --devices 0 node --backend cpu task -c result/op_cast/json/all_op_cast.json -p torch_cast.py --task accuracy -s 0
```

| 参数  | 子参数     | 说明                                                                 |
|---|---|---|
| node  | --backend  | 必选参数，表示执行后端，可选pyaclnn/cpu/npu                         |
| node  | --devices  | 当backend为pyaclnn/npu时必选，表示使用的设备id                      |
| node  | --is_compare | 是否用来做比较，可选True/False                                         |
| task  | -c         | 表示待测试的用例json文件                                              |
| task  | --task     | 表示执行的任务类型，可选accuracy/accuracy_dc/performance_device，分别表示精度比对/确定性计算/device性能，多个任务以逗号隔开 |
| task  | -s         | 表示执行的起始用例id                                                  |
| task  | -e         | 表示执行的结束用例id（不包含）                                         |
| task  | -p         | 入参为文件路径，表示自定义标杆执行逻辑。同时在用例生成时要将yaml文件中的api_type字段修改为自定义标杆文件的注册器名称 |

### 确定性计算

```bash
atk node --backend pyaclnn --devices 0 node --backend cpu --is_compare False task -c result/op_cast/json/all_op_cast.json -p torch_cast.py --task accuracy_dc -s 0 -e 1400 -rn 50
```

### device性能


#### aclnn VS npu 性能比对

进行aclnnCast算子和npu上的torch.cast算子的性能比对

```bash
atk node --backend pyaclnn --devices 0 node --backend npu --devices 0 task -c result/op_cast/json/all_op_cast.json -p torch_cast.py --task performance_device -s 0
```

#### aclnn VS aclnn 性能比对

用于将算子进行优化后，与优化前性能对比。

1. 生成优化前算子性能

```bash
atk node --backend pyaclnn --devices 0 --name aclnn_base node --backend cpu task -c result/op_cast/json/all_op_cast.json -p torch_cast.py --task performance_device -s 0
```

在`atk_output/`会生成对应的测试报告，可以用于之后的测试对比。

2. 对优化后的算子进行测试，并与之前的算子性能进行对比

```bash
atk node --backend pyaclnn --devices 0 node --backend cpu --is_compare False task -c result/op_cast/json/all_op_cast.json -p torch_cast.py --task performance_device --bm_file atk_output/all_op_cast_xxx/report/all_op_cast_reports_xxx.xlsx --bm_backend pyaclnn --bm_name aclnn_base -s 0
```

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/07/03 | 新增README |