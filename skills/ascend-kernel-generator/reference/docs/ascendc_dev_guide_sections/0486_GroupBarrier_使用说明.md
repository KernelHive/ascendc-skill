###### GroupBarrier 使用说明

当同一个 CubeResGroupHandle 中的两个 AIV 任务之间存在依赖关系时，可以使用 GroupBarrier 控制同步。假设一组 AIV A 做完任务 x 以后，另外一组 AIV B 才可以开始后续业务，称 AIV A 组为 Arrive 组，AIV B 组为 Wait 组。

基于 GroupBarrier 的组同步使用步骤如下：

1. 创建 GroupBarrier。
2. 被等待的 AIV 调用 Arrive，需要等待的 AIV 调用 Wait。

下文仅提供示例代码片段，更多完整样例请参考 GroupBarrier 样例。

## 步骤 1 创建 GroupBarrier

```cpp
constexpr int32_t ARRIVE_NUM = 2; // Arrive 组的 AIV 个数
constexpr int32_t WAIT_NUM = 6;   // Wait 组的 AIV 个数
AscendC::GroupBarrier<AscendC::PipeMode::MTE3_MODE> barA(workspace, ARRIVE_NUM, WAIT_NUM);
// 创建 GroupBarrier，用户自行管理并对这部分 workspace 清零
```

## 步骤 2 被等待的 AIV 调用 Arrive，需要等待的 AIV 调用 Wait

```cpp
auto id = AscendC::GetBlockIdx();
if (id > 0 && id < ARRIVE_NUM) {
    // 各种 Vector 计算逻辑，用户自行实现
    barA.Arrive(id);
} else if (id >= ARRIVE_NUM && id < ARRIVE_NUM + WAIT_NUM) {
    barA.Wait(id - ARRIVE_NUM);
    // 各种 Vector 计算逻辑，用户自行实现
}
```
