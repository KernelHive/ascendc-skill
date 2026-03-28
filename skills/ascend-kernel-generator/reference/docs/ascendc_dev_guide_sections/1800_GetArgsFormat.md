##### GetArgsFormat

## 函数功能
获取算子的 ArgsFormat。

## 函数原型
```cpp
const char *GetArgsFormat() const
```

## 参数说明
无

## 返回值说明
- 成功时返回算子的 ArgsFormat。
- 失败时，返回 `nullptr`。

## 约束说明
只有 AI CPU 和 AI Core 类型的任务能获取到 ArgsFormat。

## 调用示例
```cpp
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
                               std::vector<std::vector<uint8_t>> &tasks) {
    // ...
    auto aicore_task = KernelLaunchInfo::LoadFromData(context, tasks.back());
    auto aicore_args_format_str = aicore_task.GetArgsFormat();
    auto aicore_args_format = ArgsFormatSerializer::Deserialize(aicore_args_format_str);
    // ...
}
```
