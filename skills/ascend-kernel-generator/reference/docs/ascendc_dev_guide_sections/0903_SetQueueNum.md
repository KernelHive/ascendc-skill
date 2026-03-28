###### SetQueueNum

## 功能说明

设置每个向服务端下发任务的核上的 BatchWrite 通信队列数量。

## 函数原型

```cpp
uint32_t SetQueueNum(uint16_t num)
```

## 参数说明

**表 15-913 参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| num    | 输入      | 表示队列的数量。参与通信的核数 × 队列数量支持设置的取值范围为 [0, 40]，参与通信的核数的设置请参考 `SetCommBlockNum`。 |

## 返回值说明

- 0 表示设置成功。
- 非 0 表示设置失败。

## 约束说明

本接口仅在 Atlas A3 训练系列产品 / Atlas A3 推理系列产品上通信类型为 `HCCL_CMD_BATCH_WRITE` 时生效。

## 调用示例

```cpp
const char *groupName = "testGroup";
uint32_t opType = HCCL_CMD_BATCH_WRITE;
std::string algConfig = "BatchWrite=level0:fullmesh";
uint32_t reduceType = HCCL_REDUCE_SUM;
AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType);
mc2CcTilingConfig.SetQueueNum(2U);
```
