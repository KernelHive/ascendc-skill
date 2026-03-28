###### v1 版本 TilingData（废弃）

## 说明

该结构体已废弃，并将在后续版本移除，请不要使用该结构体。无需直接对该结构体中的成员进行设置，统一使用 Hccl Tiling 提供的接口设置即可。

对于本节介绍的 TilingData 结构体，当构建通信计算融合算子时，通算融合算子的 TilingData 结构体中，计算 Tiling 结构体部分必须在本节的通信 Tiling 结构体后追加。

对于 v1 和 v2 两个版本的 TilingData，Tiling 结构体的第一个 uint32_t 字段用于区分两个版本，即 v1 版本的 preparePosition 字段，v2 版本的 version 字段。若使用 v2 版本的 Tiling 结构体，则必须设置 version=2；若使用 v1 版本的 Tiling 结构体，则设置 preparePosition=1。用户使用任意版本的 TilingData 时，都必须严格按照对应版本的 Tiling 结构体，将其作为算子 TilingData 结构体的组成部分。

## 功能说明

AI CPU 启动下发通信任务前，需获取固定的通信配置 Mc2Msg。在算子实现中，由 Tiling 组装通信配置项，通过配置固定参数和固定参数顺序的 Tiling Data，将通信配置信息在调用 AI CPU 通信接口时传递给 AI CPU。

## 参数说明

**表 15-914 Mc2Msg 参数说明**

| 参数名 | 描述 |
|--------|------|
| preparePosition | 设置服务端组装任务的方式，用户需要在 Tiling 中显式赋值，uint32_t 类型，当前支持的取值如下：<br>1：AI CPU 与 AI Core 通过通信任务机制实现消息传递和任务下发；由 AI Core 侧通过消息通知时设置为 1，即算子中使用 Hccl 时设置为 1。 |
| sendOff | 预留参数，不可配置。 |
| recvOff | 预留参数，不可配置。 |
| tailSendOff | 预留参数，不可配置。 |
| tailRecvOff | 预留参数，不可配置。 |
| sendCnt | 预留参数，不可配置。 |
| recvCnt | 预留参数，不可配置。 |
| tailSendCnt | 预留参数，不可配置。 |
| tailRecvCnt | 预留参数，不可配置。 |
| totalCnt | 预留参数，不可配置。 |
| turnNum | 预留参数，不可配置。 |
| tailNum | 预留参数，不可配置。 |
| stride | 预留参数，不可配置。 |
| workspaceOff | 预留参数，不可配置。 |
| notifyOff | 预留参数，不可配置。 |
| notifyBeginCnt | 预留参数，不可配置。 |
| notifyEndCnt | 预留参数，不可配置。 |
| useBufferType | 设置通信算法获取输入数据的位置，uint8_t 类型，参数取值如下：<br>• 0：默认值，默认通信输入不放在 windows 中，其中 windows 为其他卡可访问的共享缓冲区。<br>• 1：通信输入不放在 windows 中，当前该参数取值 1 与取值 0 的功能一致。<br>• 2：通信输入放在 windows 中，仅适用于 AllReduce 算法。 |
| funID | 预留参数，不可配置。 |
| dataType | 预留参数，不可配置。 |
| groupNum | 预留参数，不可配置。 |
| reuseMode | 预留参数，不可配置。 |
| commType | 预留参数，不可配置。 |
| reduceOp | 预留参数，不可配置。 |
| commOrder | 预留参数，不可配置。 |
| waitPolicy | 预留参数，不可配置。 |
| rspPolicy | 预留参数，不可配置。 |
| exitPolicy | 预留参数，不可配置。 |
| commAlg | 设置具体通信算法，用户需要在 Tiling 中显式赋值，uint8_t 类型，当前支持的取值如下：<br>1：FullMesh 算法，即 NPU 之间的全连接，任意两个 NPU 之间可以直接进行数据收发。详细的算法内容可参见《HCCL 集合通信库用户指南》中的集合通信算法章节。 |
| taskType | 预留参数，不可配置。 |
| debugMode | 预留参数，不可配置。 |
| stepSize | 预留参数，不可配置。 |
| sendArgIndex | 预留参数，不可配置。 |
| recvArgIndex | 预留参数，不可配置。 |
| commOutArgIndex | 预留参数，不可配置。 |
| hasCommOut | 本卡的通信算法的计算结果是否输出到 recvBuf（目的数据 buffer 地址）。仅 AllGather 算法与 AlltoAll 算法支持配置该参数。uint8_t 类型，参数取值如下：<br>• 0：不输出本卡通信算法的计算结果。在无需输出通信结果时，配置参数值为 0，此时不会拷贝本卡的通信结果数据，可提升算子性能。例如，在 8 卡场景下，本卡只取其他卡的部分数据，这时可配置本参数为 0。<br>• 1：输出本卡通信算法的计算结果。 |
| reserve | 保留字段。 |
| reserve2 | 保留字段。 |

## 约束说明

- 算子的 Tiling Data 结构需要按顺序完整包含 Mc2Msg 参数。
- AI CPU 需获取固定数据结构的通信配置，算子注册 Tiling Data 时保持该结构的一致性。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品暂不支持该版本 TilingData。

## 调用示例

以自定义算子 AllGatherMatmulCustom 为例，如下为该算子的算子原型，"gather_out" 为通信任务 AllGather 的输出。

```json
[
  {
    "op": "AllGatherMatmulCustom",
    "input_desc": [
      {
        "name": "x1",
        "param_type": "required",
        "format": [
          "ND",
          "ND"
        ],
        "type": [
          "float16",
          "bfloat16"
        ]
      },
      {
        "name": "x2",
        "param_type": "required",
        "format": [
          "ND",
          "ND"
        ],
        "type": [
          "float16",
          "bfloat16"
        ]
      },
      {
        "name": "bias",
        "param_type": "optional",
        "format": [
          "ND",
          "ND"
        ],
        "type": [
          "float16",
          "bfloat16"
        ]
      }
    ],
    "output_desc": [
      {
        "name": "y",
        "param_type": "required",
        "format": [
          "ND",
          "ND"
        ],
        "type": [
          "float16",
          "bfloat16"
        ]
      },
      {
        "name": "gather_out",
        "param_type": "required",
        "format": [
          "ND",
          "ND"
        ],
        "type": [
          "float16",
          "bfloat16"
        ]
      }
    ],
    "attr": [
      {
        "name": "group",
        "type": "string",
        "default_value": "",
        "param_type": "required"
      },
      {
        "name": "rank_size",
        "type": "int",
        "default_value": 0,
        "param_type": "optional"
      },
      {
        "name": "is_gather_out",
        "type": "bool",
        "default_value": true,
        "param_type": "optional"
      }
    ]
  }
]
```

算子的 Tiling Data 结构需要按顺序完整包含 Mc2Msg 参数，如下为算子 Tiling Data 代码示例。

```cpp
// 声明 Mc2Msg 结构
BEGIN_TILING_DATA_DEF(Mc2Msg)
TILING_DATA_FIELD_DEF(uint32_t, preparePosition);
TILING_DATA_FIELD_DEF(uint32_t, sendOff);
TILING_DATA_FIELD_DEF(uint32_t, recvOff);
TILING_DATA_FIELD_DEF(uint32_t, tailSendOff);
TILING_DATA_FIELD_DEF(uint32_t, tailRecvOff);
TILING_DATA_FIELD_DEF(uint64_t, sendCnt);
TILING_DATA_FIELD_DEF(uint32_t, recvCnt);
TILING_DATA_FIELD_DEF(uint32_t, tailSendCnt);
TILING_DATA_FIELD_DEF(uint32_t, tailRecvCnt);
TILING_DATA_FIELD_DEF(uint32_t, totalCnt);
TILING_DATA_FIELD_DEF(uint32_t, turnNum);
TILING_DATA_FIELD_DEF(uint32_t, tailNum);
TILING_DATA_FIELD_DEF(uint32_t, stride);
TILING_DATA_FIELD_DEF(uint32_t, workspaceOff);
TILING_DATA_FIELD_DEF(uint32_t, notifyOff);
TILING_DATA_FIELD_DEF(uint16_t, notifyBeginCnt);
TILING_DATA_FIELD_DEF(uint16_t, notifyEndCnt);
TILING_DATA_FIELD_DEF(uint8_t, useBufferType);
TILING_DATA_FIELD_DEF(uint8_t, funID);
TILING_DATA_FIELD_DEF(uint8_t, dataType);
TILING_DATA_FIELD_DEF(uint8_t, groupNum);
TILING_DATA_FIELD_DEF(uint8_t, reuseMode);
TILING_DATA_FIELD_DEF(uint8_t, commType);
TILING_DATA_FIELD_DEF(uint8_t, reduceOp);
TILING_DATA_FIELD_DEF(uint8_t, commOrder);
TILING_DATA_FIELD_DEF(uint8_t, waitPolicy);
TILING_DATA_FIELD_DEF(uint8_t, rspPolicy);
TILING_DATA_FIELD_DEF(uint8_t, exitPolicy);
TILING_DATA_FIELD_DEF(uint8_t, commAlg);
TILING_DATA_FIELD_DEF(uint8_t, taskType);
TILING_DATA_FIELD_DEF(uint8_t, debugMode);
TILING_DATA_FIELD_DEF(uint8_t, stepSize);
TILING_DATA_FIELD_DEF(uint8_t, sendArgIndex);
TILING_DATA_FIELD_DEF(uint8_t, recvArgIndex);
TILING_DATA_FIELD_DEF(uint8_t, commOutArgIndex);
TILING_DATA_FIELD_DEF(uint8_t, hasCommOut);
TILING_DATA_FIELD_DEF(uint8_t, reserve);
TILING_DATA_FIELD_DEF(uint32_t, reserve2);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Mc2MsgOp, Mc2Msg)

BEGIN_TILING_DATA_DEF(AllGatherMatmulCustomTilingData)
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
END_TILING_DATA_DEF;

// 配置 Mc2Msg
AllGatherMatmulCustomTilingData tiling;
tiling.msg.set_preparePosition(1);
tiling.msg.set_commAlg(1);
tiling.msg.set_useBufferType(1);
tiling.msg.set_hasCommOut(1);
```
