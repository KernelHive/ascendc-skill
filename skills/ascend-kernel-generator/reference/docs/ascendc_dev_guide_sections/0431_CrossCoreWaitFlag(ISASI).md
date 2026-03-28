###### CrossCoreWaitFlag(ISASI)

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品 AI Core | x |
| Atlas 推理系列产品 Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

面向分离模式的核间同步控制接口。该接口和 `CrossCoreSetFlag` 接口配合使用。具体使用方法请参考 `CrossCoreSetFlag`。

## 函数原型

```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
```

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| modeId | 核间同步的模式，取值如下：<br>● 模式0：AI Core 核间的同步控制。<br>● 模式1：AI Core 内部，Vector 核（AIV）之间的同步控制。<br>● 模式2：AI Core 内部，Cube 核（AIC）与 Vector 核（AIV）之间的同步控制。 |
| pipe | 设置这条指令所在的流水类型，流水类型可参考硬件流水类型。 |

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| flagId | 输入 | 核间同步的标记。<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，取值范围是 0-10。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，取值范围是 0-10。 |

## 返回值说明

无

## 约束说明

- 使用该同步接口时，需要按照如下规则设置 Kernel 类型：
  - 在纯 Vector/Cube 场景下，需设置 Kernel 类型为 `KERNEL_TYPE_MIX_AIV_1_0` 或 `KERNEL_TYPE_MIX_AIC_1_0`。
  - 对于 Vector 和 Cube 混合场景，需根据实际情况灵活配置 Kernel 类型。
- `CrossCoreWaitFlag` 必须与 `CrossCoreSetFlag` 接口配合使用，避免计算核一直处于阻塞阶段。
- 如果执行 `CrossCoreWaitFlag` 时该 `flagId` 的计数器的值为 0，则 `CrossCoreWaitFlag` 之后的所有指令都将被阻塞，直到该 `flagId` 的计数器的值不为 0。同一个 `flagId` 的计数器最多设置 15 次。

## 调用示例

请参考调用示例。
