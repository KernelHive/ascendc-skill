##### CreateHcomRecordTask

## 函数功能

创建一个 Record task，此 Task 用于唤醒其他流上的与其有相同 group_name 的 Wait Task。

## 函数原型

```cpp
static KernelLaunchInfo CreateHcomRecordTask(const gert::ExeResGenerationContext *context, const char *group_name = "group")
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| context | 输入 | GenerateTask 函数的入参，保存了算子的基础信息。 |
| group_name | 输入 | Record task 的分组名字，默认为 group，用于与 Wait Task 配套。 |

## 返回值说明

返回创建出来的 Record Task 信息。

## 约束说明

group_name 必须与算子原型中定义的属性一致。例如，某个 mc2 算子定义了一个属性 group_ep，则可以使用 group_name 为 group_ep 创建 Record 任务和 Wait 任务。

## 调用示例

```cpp
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
                               std::vector<std::vector<uint8_t>> &tasks) {
    ...
    // 创建 RecordTask
    auto record_task = KernelLaunchInfo::CreateHcomRecordTask(context);
    // 设置流信息
    record_task.SetStreamId(stream_id);
    tasks.insert(tasks.begin() + aicore_index, record_task.Serialize());
    aicore_index++;
    ...
}
```
