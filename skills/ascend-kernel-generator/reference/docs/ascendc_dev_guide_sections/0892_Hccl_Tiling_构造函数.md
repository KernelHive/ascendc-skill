###### Hccl Tiling 构造函数

## 功能说明

用于创建一个 `Mc2CcTilingConfig` 对象。

## 函数原型

```cpp
Mc2CcTilingConfig(const std::string &groupName, uint32_t opType, const std::string &algConfig, uint32_t reduceType = 0, uint8_t dstDataType = 0, uint8_t srcDataType = 0)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| groupName | 输入 | 当前通信任务所在的通信域。string类型，支持的最大长度为128字节。 |
| opType | 输入 | 表示通信任务类型。uint32_t类型。Hccl API提供 `HcclCMDType` 枚举定义作为该参数的取值，具体支持的通信任务类型及取值请参考表15-902。 |
| algConfig | 输入 | 通信算法配置。string类型，支持的最大长度为128字节。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，该参数为预留字段，配置后不生效，默认仅支持FullMesh算法。FullMesh算法即NPU之间的全连接，任意两个NPU之间可以直接进行数据收发。详细的算法内容可参见《HCCL集合通信库用户指南》中的集合通信算法章节。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，当前支持的取值为：<br>• `"AllReduce=level0:doublering"`：AllReduce通信任务<br>• `"AllGather=level0:doublering"`：AllGather通信任务<br>• `"ReduceScatter=level0:doublering"`：ReduceScatter通信任务<br>• `"AlltoAll=level0:fullmesh;level1:pairwise"`：AlltoAllV和AlltoAll通信任务<br>• `"BatchWrite=level0:fullmesh"`：BatchWrite通信任务 |
| reduceType | 输入 | 归约操作类型，仅对有归约操作的通信任务生效。uint32_t类型，取值详见表15-870。 |
| dstDataType | 输入 | 通信任务中输出数据的数据类型。uint8_t类型，该参数的取值范围请参考表15-869。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，该参数暂不支持，配置后不生效。<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，该参数暂不支持，配置后不生效。 |
| srcDataType | 输入 | 通信任务中输入数据的数据类型。uint8_t类型，该参数的取值范围请参考表15-869。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，该参数暂不支持，配置后不生效。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，该参数暂不支持，配置后不生效。 |

### HcclCMDType 参数说明

**数据类型说明**

`HcclCMDType` 通信任务类型。

- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，当前支持的通信任务类型为：
  `HCCL_CMD_ALLREDUCE`、`HCCL_CMD_ALLGATHER`、`HCCL_CMD_REDUCE_SCATTER`、`HCCL_CMD_ALLTOALL`、`HCCL_CMD_BATCH_WRITE`

- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，当前支持的通信任务类型为：
  `HCCL_CMD_ALLREDUCE`、`HCCL_CMD_ALLGATHER`、`HCCL_CMD_REDUCE_SCATTER`、`HCCL_CMD_ALLTOALL`、`HCCL_CMD_ALLTOALLV`、`HCCL_CMD_BATCH_WRITE`

```cpp
enum class HcclCMDType {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_GATHER,
    HCCL_CMD_SCATTER,
    HCCL_CMD_BATCH_SEND_RECV,
    HCCL_CMD_BATCH_PUT,
    HCCL_CMD_BATCH_GET,
    HCCL_CMD_ALLGATHER_V,
    HCCL_CMD_REDUCE_SCATTER_V,
    HCCL_CMD_BATCH_WRITE,
    HCCL_CMD_ALL,
    HCCL_CMD_HALF_ALLTOALLV,
    // control task start from enum value 100, reserving for comm tasks
    HCCL_CMD_FINALIZE = 100,
    HCCL_CMD_INTER_GROUP_SYNC,
    HCCL_CMD_MAX
};
```

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
const char *groupName = "testGroup";
uint32_t opType = HCCL_CMD_REDUCE_SCATTER;
std::string algConfig = "ReduceScatter=level0:fullmesh";
uint32_t reduceType = HCCL_REDUCE_SUM;
AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType); // 构造函数
mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling); // tiling为算子组装的TilingData结构体
mc2CcTilingConfig.GetTiling(tiling->reduceScatterTiling);
```
