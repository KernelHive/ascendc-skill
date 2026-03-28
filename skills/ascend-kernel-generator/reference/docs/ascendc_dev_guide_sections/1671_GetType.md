##### GetType

## 函数功能
获取当前 ArgDescInfo 的类型。

## 函数原型
```cpp
ArgDescType GetType() const
```

## 参数说明
无

## 返回值说明
返回 ArgDescInfo 的 type，类型为 `ArgDescType`（参见 15.2.3.56 节）。

失败时返回 `kEnd`。

## 约束说明
无

## 调用示例
```cpp
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
                               std::vector<std::vector<uint8_t>> &tasks) {
    ...
    // 更改原 AI Core 任务的 argsformat
    auto aicore_task = KernelLaunchInfo::LoadFromData(context, tasks.back());
    auto aicore_args_format_str = aicore_task.GetArgsFormat();
    auto aicore_args_format = ArgsFormatSerializer::Deserialize(aicore_args_format_str);
    size_t i = 0UL;
    for (; i < aicore_args_format.size(); i++) {
        if (aicore_args_format[i].GetType() == ArgDescType::kIrInput ||
            aicore_args_format[i].GetType() == ArgDescType::kInputInstance) {
            break;
        }
    }
    aicore_args_format.insert(aicore_args_format.begin() + i,
                              ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom));
    ...
}
```
