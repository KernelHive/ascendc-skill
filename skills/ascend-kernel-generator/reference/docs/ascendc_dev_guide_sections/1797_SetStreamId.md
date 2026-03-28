##### SetStreamId

## 函数功能
设置 Task 的流 ID。

## 函数原型
```cpp
void SetStreamId(uint32_t stream_id)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| stream_id | 输入 | 流 ID。 |

## 返回值说明
无

## 约束说明
无

## 调用示例
```cpp
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
                               std::vector<std::vector<uint8_t>> &tasks) {
    // ...
    auto wait_task = KernelLaunchInfo::CreateHcomWaitTask(context);
    wait_task.SetStreamId(attach_stream_id);
    // ...
}
```
