##### GetStreamId

## 函数功能

获取当前 Task 所在流的 ID。

## 函数原型

```cpp
uint32_t GetStreamId() const
```

## 参数说明

无

## 返回值说明

返回当前 Task 所在流的 ID，默认值为 0。  
异常时，返回 `int32_max`。

## 约束说明

无

## 调用示例

```cpp
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
                               std::vector<std::vector<uint8_t>> &tasks) {
    ...
    auto aicore_task = KernelLaunchInfo::LoadFromData(context, tasks.back());
    auto stream_id = aicore_task.GetStreamId();
    ...
}
```
