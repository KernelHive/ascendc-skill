##### GetIrIndex

## 函数功能
获取当前 ArgDescInfo 的算子 IR 索引。只有当 type 为 `kIrInput`、`kIrOutput`、`kIrInputDesc`、`kIrOutputDesc` 时才可以获取到。

## 函数原型
```cpp
int32_t GetIrIndex() const
```

## 参数说明
无

## 返回值说明
当前 Args 地址对应输入/输出的 IR 索引，未设置时默认值为 -1。

## 约束说明
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
  aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kIrOutput, 0));
  // 获取 kIrOutput 类型的 ArgDescInfo 的 IR 索引，此时获取到的值为 0
  auto ir_index = aicpu_args_format.back().GetIrIndex();
  ...
}
```
