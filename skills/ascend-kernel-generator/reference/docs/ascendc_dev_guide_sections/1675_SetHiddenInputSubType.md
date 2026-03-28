##### SetHiddenInputSubType

## 函数功能

设置 ArgDescInfo 的隐藏输入地址的类型，只有 type 为 kHiddenInput 时，才能设置成功。

## 函数原型

```cpp
graphStatus SetHiddenInputSubType(HiddenInputSubType hidden_type)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| hidden_type | 输入 | 隐藏输入的类型 |

## 返回值说明

设置成功时返回 `ge::GRAPH_SUCCESS`。

关于 graphStatus 的定义，请参见 15.2.3.55 ge::graphStatus。

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
    ArgDescInfo args_info(ArgDescType::kHiddenInput);
    args_info.SetHiddenInputSubType(HiddenInputSubType::kHcom);
    aicpu_args_format.emplace_back(args_info);
    ...
}
```
