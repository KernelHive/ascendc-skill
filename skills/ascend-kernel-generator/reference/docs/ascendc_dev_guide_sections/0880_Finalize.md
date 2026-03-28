###### Finalize

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

通知服务端后续无通信任务，执行结束后退出。

## 函数原型

```cpp
template <bool sync = true>
__aicore__ inline void Finalize()
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| sync | 输入 | 是否需要等待服务端的通信完成。bool类型，参数取值如下：<br>● true：默认值，表示客户端将检测并等待最后一个通信任务完成。<br>● false：表示客户端不会等待通信任务完成，而是直接退出。<br><br>**产品支持说明：**<br>● Atlas A3 训练系列产品/Atlas A3 推理系列产品：该参数支持默认值true，仅在通信任务为BatchWrite时，支持取值为false<br>● Atlas A2 训练系列产品/Atlas A2 推理系列产品：该参数仅支持默认值true |

## 返回值说明

无

## 约束说明

- 调用本接口前确保已调用过`InitV2`和`SetCcTilingV2`接口
- 本接口在AI Cube核或者AI Vector核上调用必须与对应的Prepare接口的调用核保持一致

## 调用示例

```cpp
REGISTER_TILING_DEFAULT(ReduceScatterCustomTilingData); // ReduceScatterCustomTilingData为对应算子头文件定义的结构体
GET_TILING_DATA_WITH_STRUCT(ReduceScatterCustomTilingData, tilingData, tilingGM);
Hccl hccl;
GM_ADDR contextGM = AscendC::GetHcclContext<0>(); // AscendC自定义算子kernel中，通过此方式获取Hccl context
hccl.InitV2(contextGM, &tilingData);
hccl.SetCcTilingV2(offsetof(ReduceScatterCustomTilingData, mc2CcTiling));
if (AscendC::g_coreType == AIC) {
    HcclHandle handleId = hccl.ReduceScatter(sendBuf, recvBuf, 100, HcclDataType::HCCL_DATA_TYPE_INT8,
    HcclReduceOp::HCCL_REDUCE_SUM, 10);
    hccl.Commit(handleId); // 通知服务端可以执行上述的ReduceScatter任务
    for (uint8_t i=0; i<10; i++) {
        hccl.Wait(handleId); // 阻塞接口，需等待上述ReduceScatter任务执行完毕
    }
    hccl.Finalize<true>(); // 后续无其他通信任务，通知服务端执行上述ReduceScatter任务之后即可以退出
}
```
