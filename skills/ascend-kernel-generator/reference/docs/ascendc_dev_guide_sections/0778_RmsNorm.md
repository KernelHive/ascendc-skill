##### RmsNorm

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

实现对shape大小为[B，S，H]的输入数据的RmsNorm归一化，其计算公式如下：

```
output = (input / sqrt(mean(input^2) + epsilon)) * gamma
```

其中，γ为缩放系数，ε为防除零的权重系数。

## 函数原型

### 通过sharedTmpBuffer入参传入临时空间

```cpp
template <typename T, bool isBasicBlock = false>
__aicore__ inline void RmsNorm(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
const LocalTensor<T>& gammaLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon,
const RmsNormTiling& tiling)
```

### 接口框架申请临时空间

```cpp
template <typename T, bool isBasicBlock = false>
__aicore__ inline void RmsNorm(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
const LocalTensor<T>& gammaLocal, const T epsilon, const RmsNormTiling& tiling)
```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持接口框架申请和开发者通过sharedTmpBuffer入参传入两种方式。

- **接口框架申请临时空间**：开发者无需申请，但是需要预留临时空间的大小。
- **通过sharedTmpBuffer入参传入**：使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过RmsNorm Tiling中提供的GetRmsNormMaxMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float<br>Atlas 推理系列产品AI Core，支持的数据类型为：half、float |
| isBasicBlock | srcTensor和dstTensor的shape信息和Tiling切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。基本块要求srcTensor和dstTensor的shape需要满足如下条件：<br>• last轴即H的长度为64的倍数，但小于2048；<br>• 非last轴长度（B*S）为8的倍数。 |

### 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| dstLocal | 输出 | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>dstLocal的shape和源操作数srcLocal需要保持一致。 |
| srcLocal | 输入 | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>shape为[B, S, H]，尾轴H长度需要满足32字节对齐。 |
| gammaLocal | 输入 | 缩放系数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>shape需要与srcLocal和dstLocal的尾轴H长度相等，即shape为[H]。 |
| sharedTmpBuffer | 输入 | 临时空间。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>接口内部复杂计算时用于存储中间变量，由开发者提供。<br>临时空间大小BufferSize的获取方式请参考15.1.5.4.8 RmsNorm Tiling。 |
| epsilon | 输入 | 防除零的权重系数，数据类型需要与srcLocal/dstLocal保持一致。 |
| tiling | 输入 | RmsNorm计算所需Tiling信息，Tiling信息的获取请参考相关文档。 |
