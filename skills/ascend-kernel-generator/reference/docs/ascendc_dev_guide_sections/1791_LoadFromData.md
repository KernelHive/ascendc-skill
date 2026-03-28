##### LoadFromData

## 功能描述
提供将序列化后的 Task 任务反序列化的能力，主要用于在 `GenerateTask` 函数中，将框架构造的 Task 反序列化后修改其参数。

## 函数原型
```cpp
static KernelLaunchInfo LoadFromData(const gert::ExeResGenerationContext *context, const std::vector<uint8_t> &data)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| context | 输入 | GenerateTask 函数的入参，保存了算子的基础信息 |
| data | 输入 | KernelLaunchInfo 序列化后的结果 |

## 返回值说明
返回 data 入参反序列化后的结果。

## 约束说明
无

## 调用示例

```cpp
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
                               std::vector<std::vector<uint8_t>> &tasks) {
    // 更改原AI Core任务的ArgsFormat
    auto aicore_task = KernelLaunchInfo::LoadFromData(context, tasks.back());
    
    // 从task中获取argsformat
    auto aicore_args_format_str = aicore_task.GetArgsFormat();
    auto aicore_args_format = ArgsFormatSerializer::Deserialize(aicore_args_format_str);
    
    size_t i = 0UL;
    // 找到第一个输入地址的位置
    for (; i < aicore_args_format.size(); i++) {
        if (aicore_args_format[i].GetType() == ArgDescType::kIrInput ||
            aicore_args_format[i].GetType() == ArgDescType::kInputInstance) {
            break;
        }
    }
    
    // 在输入地址前插入一个通信地址
    aicore_args_format.insert(aicore_args_format.begin() + i,
                              ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom));
    
    // 将修改后的argsformat序列化后设置回Task中
    aicore_task.SetArgsFormat(ArgsFormatSerializer::Serialize(aicore_args_format).GetString());
    
    // 重新序列化并设置到Tasks中
    tasks.back() = aicore_task.Serialize();
    
    return SUCCESS;
}
```
