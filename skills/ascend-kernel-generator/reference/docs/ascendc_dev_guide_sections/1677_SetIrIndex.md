##### SetIrIndex

## 功能
设置当前 ArgDescInfo 的 IR 索引。只有当 type 为 `kIrInput`、`kIrOutput`、`kIrInputDesc`、`kIrOutputDesc` 时才可以设置成功。

## 函数原型
```cpp
void SetIrIndex(int32_t ir_index)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ir_index | 输入 | 输入/输出的 IR 索引 |

## 返回值
无

## 约束
无

## 调用示例
```cpp
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
                               std::vector<std::vector<uint8_t>> &tasks) {
  ...
  // 设置 AI CPU 任务
  auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(context,
                                                         "libccl_kernel.so",
                                                         "RunAicpuKfcSrvLaunch");
  std::vector<ArgDescInfo> aicpu_args_format;
  aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kIrOutput));
  // 将 IR 索引设置为 0
  aicpu_args_format.back().SetIrIndex(0);
  ...
}
```
