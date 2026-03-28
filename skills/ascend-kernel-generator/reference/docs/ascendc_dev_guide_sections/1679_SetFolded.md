##### SetFolded

## 函数功能

设置当前地址是否需要被折叠成二级指针。

## 函数原型

```cpp
void SetFolded(bool is_folded)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| is_folded | 输入 | 此地址是否需要折叠成二级指针 |

## 返回值说明

无

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
    aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kIrOutputDesc, 0));
    // 将地址设置成需要折叠成二级指针
    aicpu_args_format.back().SetFolded(true);
    ...
}
```
