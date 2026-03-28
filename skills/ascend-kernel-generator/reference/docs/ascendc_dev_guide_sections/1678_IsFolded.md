##### IsFolded

## 函数功能
判断当前 Args 地址是否需要被折叠成二级指针。

## 函数原型
```cpp
bool IsFolded() const
```

## 参数说明
无

## 返回值说明
返回当前 Args 地址是否需要被折叠成二级指针。
- `true`：需要
- `false`：不需要

## 约束说明
无

## 调用示例
```cpp
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
                               std::vector<std::vector<uint8_t>> &tasks) {
    ...
    // 设置AI CPU任务
    auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(context,
                                                           "libccl_kernel.so",
                                                           "RunAicpuKfcSrvLaunch");
    std::vector<ArgDescInfo> aicpu_args_format;
    aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kIrOutputDesc, 0, true));
    // 判断此地址是否需要被折叠成二级指针，此处返回 true
    auto is_folded = aicpu_args_format.back().IsFolded();
    ...
}
```
