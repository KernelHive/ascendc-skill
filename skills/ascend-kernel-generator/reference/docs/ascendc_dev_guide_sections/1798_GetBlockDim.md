##### GetBlockDim

## 函数功能
获取算子 BlockDim。

## 函数原型
```cpp
uint32_t GetBlockDim() const
```

## 参数说明
无

## 返回值说明
返回此算子 Task 的 BlockDim 值，默认值为 0。

异常时返回 `int32_max`。

## 约束说明
无

## 调用示例
```cpp
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
                               std::vector<std::vector<uint8_t>> &tasks) {
    ...
    auto aicore_task = KernelLaunchInfo::LoadFromData(context, tasks.back());
    auto block_dim = aicore_task.GetBlockDim();
    ...
}
```
