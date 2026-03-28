##### Deserialize

## 函数功能

ArgsFormat 的反序列化接口。

## 函数原型

```cpp
static std::vector<ArgDescInfo> Deserialize(const AscendString &args_str)
```

## 参数说明

| 参数名   | 输入/输出 | 描述                         |
|----------|-----------|------------------------------|
| args_str | 输入      | ArgsFormat 的序列化字符串。 |

## 返回值

- 成功：返回 ArgsFormat
- 失败：返回空的 vector

## 约束说明

字符串必须是 ArgsFormat 的序列化结果。

## 调用示例

```cpp
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
                               std::vector<std::vector<uint8_t>> &tasks) {
    // ...
    // 更改原AI Core任务的ArgsFormat
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
    aicore_task.SetArgsFormat(ArgsFormatSerializer::Serialize(aicore_args_format).GetString());
    tasks.back() = aicore_task.Serialize();
    return SUCCESS;
}
```
