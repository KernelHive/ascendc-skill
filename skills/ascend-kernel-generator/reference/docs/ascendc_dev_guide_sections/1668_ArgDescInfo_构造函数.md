##### ArgDescInfo 构造函数

## 函数功能
ArgDescInfo 构造函数。

## 函数原型
```cpp
explicit ArgDescInfo(ArgDescType arg_type, int32_t ir_index = -1, bool is_folded = false)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| arg_type | 输入 | 当前 Args 地址的类型。具体类型定义请参考 15.2.3.56 ArgDescType。 |
| ir_index | 输入 | 当前 Args 地址对应的算子 IR 索引。 |
| is_folded | 输入 | 当前地址是否需要被折叠成二级指针。 |

## 返回值说明
无

## 约束说明
无

## 调用示例
```cpp
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
                               std::vector<std::vector<uint8_t>> &tasks) {
    ...
    // 设置 AI CPU 任务
    auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(context, "libccl_kernel.so",
                                                           "RunAicpuKfcSrvLaunch");
    std::vector<ArgDescInfo> aicpu_args_format;
    // 构造了一个类型为 kIrOutputDesc，ir_index 为 0，需要被折叠成二级指针的地址描述信息
    aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kIrOutputDesc, 0, true));
    ...
}
```
