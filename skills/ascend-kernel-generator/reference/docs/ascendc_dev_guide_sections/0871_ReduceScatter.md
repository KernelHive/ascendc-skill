###### ReduceScatter

```markdown
## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 AI Core | × |
| Atlas 推理系列产品 Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

集合通信算子 ReduceScatter 的任务下发接口，返回该任务的标识 handleId 给用户。

ReduceScatter 的功能为：将所有 rank 的输入相加（或其他归约操作）后，再把结果按照 rank 编号均匀分散到各个 rank 的输出 buffer，每个进程拿到其他进程 1/ranksize 份的数据进行归约操作。

## 函数原型

```cpp
template <bool commit = false>
__aicore__ inline HcclHandle ReduceScatter(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, uint64_t strideCount, uint8_t repeat = 1)
```

## 参数说明

### 模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| commit | 输入 | bool 类型。参数取值如下：<br>● true：在调用 Prepare 接口时，Commit 同步通知服务端可以执行该通信任务。<br>● false：在调用 Prepare 接口时，不通知服务端执行该通信任务。 |

### 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| sendBuf | 输入 | 源数据 buffer 地址。 |
| recvBuf | 输出 | 目的数据 buffer 地址，集合通信结果输出到此 buffer 中。 |
| recvCount | 输入 | 参与 ReduceScatter 操作的 recvBuf 的数据个数；sendBuf 的数据个数等于 recvCount * rank size。 |
| dataType | 输入 | ReduceScatter 操作的数据类型，目前支持 float、half、int8_t、int16_t、int32_t、bfloat16_t 数据类型，即支持取值为 HCCL_DATA_TYPE_FP32、HCCL_DATA_TYPE_FP16、HCCL_DATA_TYPE_INT8、HCCL_DATA_TYPE_INT16、HCCL_DATA_TYPE_INT32、HCCL_DATA_TYPE_BFP16。HcclDataType 数据类型的介绍请参考表 15-869。 |
| op | 输入 | ReduceScatter 的操作类型，目前支持 sum、max、min 操作类型，即支持取值为 HCCL_REDUCE_SUM、HCCL_REDUCE_MAX、HCCL_REDUCE_MIN。HcclReduceOp 数据类型的介绍请参考表 15-870。 |
| strideCount | 输入 | 当将一张卡上 sendBuf 中的数据 scatter 到多张卡的 recvBuf 时，需要用 strideCount 参数表示 sendBuf 上相邻数据块间的起始地址的偏移量。<br>● strideCount=0，表示从当前卡发送数据给其它卡时，相邻数据块保持地址连续。本卡发送数据到卡 rank[i]，且本卡数据块在 sendBuf 中的偏移为 i*recvCount。非多轮切分场景下，推荐用户设置该参数为 0。<br>● strideCount>0，表示从当前卡发送数据给其它卡时，相邻数据块在 sendBuf 中起始地址的偏移数据量为 strideCount。本卡发送数据到卡 rank[i]，且本卡数据块在 SendBuf 中的偏移为 i*strideCount。<br>注意：上述的偏移数据量为数据个数，单位为 sizeof(dataType)。 |
| repeat | 输入 | 一次下发的 ReduceScatter 通信任务个数。repeat 取值 ≥1，默认值为 1。当 repeat>1 时，每个 ReduceScatter 任务的 sendBuf 和 recvBuf 地址由服务端自动算出，计算公式如下：<br>sendBuf[i] = sendBuf + recvCount * sizeof(datatype) * i, i∈[0, repeat)<br>recvBuf[i] = recvBuf + recvCount * sizeof(datatype) * i, i∈[0, repeat)<br>注意：当设置 repeat>1 时，须与 strideCount 参数配合使用，规划通信数据地址。 |

## 通信示例

以上图为例，假设 4 张卡的场景，每份数据被切分为 3 块（TileCnt 为 3），每张卡上的 0-0、0-1、0-2 数据最终 reduce+scatter 到卡 rank0 的 recvBuf 上，其余的每块 1-y、2-y、3-y 数据类似，最终分别 reduce+scatter 到卡 rank1、rank2 和 rank3 的 recvBuf 上。因此，对一张卡上的数据需要调用 3 次 ReduceScatter 接口，完成每份数据的 3 块切分数据的通信。对于每一份数据，本接口中参数 recvCount 为 TileLen，strideCount 为 TileLen*TileCnt（即数据块 0-0 和 1-0 间隔的数据个数）。由于本例为内存连续场景，因此也可以只调用 1 次 ReduceScatter 接口，并将 repeat 参数设置为 3。

## 返回值说明

返回该任务的标识 handleId，handleId 大于等于 0。调用失败时，返回 -1。

## 约束说明

● 调用本接口前确保已调用过 InitV2 和 SetCcTilingV2 接口。
● 若 Hccl 对象的 config 模板参数未指定下发通信任务的核，该接口只能在 AI Cube 核或者 AI Vector 核两者之一上调用。若 Hccl 对象的 config 模板参数中指定了下发通信任务的核，则该接口可以在 AI Cube 核和 AI Vector 核上同时调用，接口内部会根据指定的核的类型，只在 AI Cube 核、AI Vector 核二者之一下发该通信任务。
● 对于 Atlas A2 训练系列产品/Atlas A2 推理系列产品，一个通信域内，所有 Prepare 接口的总调用次数不能超过 63。
● 对于 Atlas A3 训练系列产品/Atlas A3 推理系列产品，一个通信域内，所有 Prepare 接口和 InterHcclGroupSync 接口的总调用次数不能超过 63。

## 调用示例

### 非多轮切分场景

如下图所示，4 张卡上均有 300 * 4=1200 个 float16 数据，每张卡从 xGM 内存中获取到本卡数据，对各卡数据完成 reduce sum 计算后的结果数据，进行 scatter 处理，最终每张卡都得到 300 个 reduce sum 后的 float16 数据。

```cpp
extern "C" __global__ __aicore__ void reduce_scatter_custom(GM_ADDR xGM, GM_ADDR yGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM) {
    auto sendBuf = xGM; // xGM为ReduceScatter的输入GM地址
    auto recvBuf = yGM; // yGM为ReduceScatter的输出GM地址
    uint64_t recvCount = 300; // 每张卡的通信结果数据个数
    uint64_t strideCount = 0; // 非切分场景strideCount可设置为0
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    REGISTER_TILING_DEFAULT(ReduceScatterCustomTilingData); //ReduceScatterCustomTilingData为对应算子头文件定义的结构体
    GET_TILING_DATA_WITH_STRUCT(ReduceScatterCustomTilingData, tilingData, tilingGM);

    Hccl hccl;
    GM_ADDR contextGM = AscendC::GetHcclContext<0>(); // AscendC自定义算子kernel中，通过此方式获取Hccl context
    if (AscendC::g_coreType == AIV) { // 指定AIV核通信
        hccl.InitV2(contextGM, &tilingData);
        auto ret = hccl.SetCcTilingV2(offsetof(ReduceScatterCustomTilingData, reduceScatterCcTiling));
        if (ret != HCCL_SUCCESS) {
            return;
        }
        HcclHandle handleId1 = hccl.ReduceScatter<true>(sendBuf, recvBuf, recvCount,
            HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp, strideCount);
        hccl.Wait(handleId1);
        AscendC::SyncAll<true>(); // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死
        hccl.Finalize();
    }
}
```

### 多轮切分场景

使能多轮切分，等效处理上述非多轮切分示例的通信。如下图所示，每张卡的每份 300 个 float16 数据，被切分为 2 个首块，1 个尾块。每个首块的数据量 tileLen 为 128 个 float16 数据，尾块的数据量 tailLen 为 44 个 float16 数据。在算子内部实现时，需要对切分后的数据分 3 轮进行 ReduceScatter 通信任务，将等效上述非多轮切分的通信结果。

具体实现为，第 1 轮通信，每个 rank 上的 0-0\1-0\2-0\3-0 数据块进行 ReduceScatter 处理。第 2 轮通信，每个 rank 上 0-1\1-1\2-1\3-1 数据块进行 ReduceScatter 处理。第 3 轮通信，每个 rank 上 0-2\1-2\2-2\3-2 数据块进行 ReduceScatter 处理。每一轮通信的输入数据中，各卡上相邻数据块的起始地址间隔的数据个数为 strideCount，以第一轮通信结果为例，rank0 的 0-0 数据块和 1-0 数据块，或者 1-0 数据块和 2-0 数据块，两个相邻数据块起始地址间隔的数据量 strideCount = 2*tileLen+1*tailLen=300。

```cpp
extern "C" __global__ __aicore__ void reduce_scatter_custom(GM_ADDR xGM, GM_ADDR yGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM) {
    constexpr uint32_t tileNum = 2U; // 首块数量
    constexpr uint64_t tileLen = 128U; // 首块数据个数
    constexpr uint32_t tailNum = 1U; // 尾块数量
    constexpr uint64_t tailLen = 44U; // 尾块数据个数
    auto sendBuf = xGM; // xGM为ReduceScatter的输入GM地址
    auto recvBuf = yGM; // yGM为ReduceScatter的输出GM地址
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint64_t strideCount = tileLen * tileNum + tailLen * tailNum;
    REGISTER_TILING_DEFAULT(ReduceScatterCustomTilingData); //ReduceScatterCustomTilingData为对应算子头文件定义的结构体
    GET_TILING_DATA_WITH_STRUCT(ReduceScatterCustomTilingData, tilingData, tilingGM);

    Hccl hccl;
    GM_ADDR contextGM = AscendC::GetHcclContext<0>(); // AscendC自定义算子kernel中，通过此方式获取Hccl context
    if (AscendC::g_coreType == AIV) { // 指定AIV核通信
        hccl.InitV2(contextGM, &tilingData);
        auto ret = hccl.SetCcTilingV2(offsetof(ReduceScatterCustomTilingData, reduceScatterCcTiling));
        if (ret != HCCL_SUCCESS) {
            return;
        }
        // 2个首块处理
        constexpr uint32_t tileRepeat = tileNum;
        // 除了sendBuf和recvBuf入参不同，处理2个首块的其余参数相同。故使用repaet=2，第2个首块ReduceScatter任务的sendBuf、recvBuf将由API内部自行更新
        HcclHandle handleId1 = hccl.ReduceScatter<true>(sendBuf, recvBuf, tileLen,
            HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp, strideCount, tileRepeat);
        // 1个尾块处理
        constexpr uint32_t kSizeOfFloat16 = 2U;
        sendBuf += tileLen * tileNum * kSizeOfFloat16;
        recvBuf += tileLen * tileNum * kSizeOfFloat16;
        constexpr uint32_t tailRepeat = tailNum;
        HcclHandle handleId2 = hccl.ReduceScatter<true>(sendBuf, recvBuf, tailLen,
            HcclDataType::HCCL_DATA_TYPE_FP16, reduceOp, strideCount, tailRepeat);

        for (uint8_t i=0; i<tileRepeat; i++) {
            hccl.Wait(handleId1);
        }
        hccl.Wait(handleId2);
        AscendC::SyncAll<true>(); // 全AIV核同步，防止0核执行过快，提前调用hccl.Finalize()接口，导致其他核Wait卡死
        hccl.Finalize();
    }
}
```
