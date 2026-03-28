###### NotifyNextBlock

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品 AI Core | √ |
| Atlas 推理系列产品 Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

通过写 GM 地址，通知下一个核当前核的操作已完成，下一个核可以进行操作。使用接口前，请确保已经调用 `InitDetermineComputeWorkspace` 接口，初始化共享内存。

## 函数原型

```cpp
__aicore__ inline void NotifyNextBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)
```

## 参数说明

**表 15-404 接口参数说明**

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| gmWorkspace | 输入 | 临时空间，通过写 gmWorkspace 通知其他核当前核已执行完成，其他核可以继续往下执行，类型为 GlobalTensor。 |
| ubWorkspace | 输入 | 临时空间，用于操作 gmWorkspace，类型为 LocalTensor。 |

## 返回值说明

无

## 约束说明

- 需要保证每个核调用该接口的次数相同。
- gmWorkspace 申请的空间最少要求为：`blockNum * 32Bytes`；ubWorkspace 申请的空间最少要求为：`blockNum * 32 + 32Bytes`；其中 blockNum 为调用的核数，可调用 `15.1.4.6.1 GetBlockNum` 获取。
- 分离模式下，使用该接口进行多核同步时，仅对 AIV 核生效，WaitPreBlock 和 NotifyNextBlock 之间仅支持插入矢量计算相关指令，对矩阵计算相关指令不生效。
- 使用该接口进行多核控制时，算子调用时指定的逻辑 blockDim 必须保证不大于实际运行该算子的 AI 处理器核数，否则框架进行多轮调度时会插入异常同步，导致 Kernel“卡死”现象。

## 调用示例

请参考调用示例。
