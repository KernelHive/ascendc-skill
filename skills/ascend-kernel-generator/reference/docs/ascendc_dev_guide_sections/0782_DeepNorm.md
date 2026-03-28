##### DeepNorm

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

在深层神经网络训练过程中，执行层LayerNorm归一化时，可以使用DeepNorm进行替代，通过扩大残差连接来提高Transformer的稳定性。

本接口实现了对shape大小为[B, S, H]的输入数据的DeepNorm归一化，其计算公式如下：

```
DeepNorm(x) = LayerNorm(α * X + SubLayer(X))
```

SubLayer(X)通常是指在DeepNorm模型中的一个子层（sub-layer），用于实现自注意力机制（self-attention mechanism）。本接口中会整体作为一个输入Tensor传入。

其中LayerNorm的计算公式请参考LayerNorm。

## 函数原型

### 通过sharedTmpBuffer入参传入临时空间

```cpp
template <typename T, bool isReuseSrc = false, bool isBasicBlock = false>
__aicore__ inline void DeepNorm(const LocalTensor<T>& dstLocal, 
                               const LocalTensor<T>& meanLocal,
                               const LocalTensor<T>& rstdLocal, 
                               const LocalTensor<T>& srcLocal, 
                               const LocalTensor<T>& gxLocal,
                               const LocalTensor<T>& betaLocal, 
                               const LocalTensor<T>& gammaLocal, 
                               const LocalTensor<uint8_t>& sharedTmpBuffer,
                               const T alpha, 
                               const T epsilon, 
                               DeepNormTiling& tiling)
```

### 接口框架申请临时空间

```cpp
template <typename T, bool isReuseSrc = false, bool isBasicBlock = false>
__aicore__ inline void DeepNorm(const LocalTensor<T>& dstLocal, 
                               const LocalTensor<T>& meanLocal,
                               const LocalTensor<T>& rstdLocal, 
                               const LocalTensor<T>& srcLocal, 
                               const LocalTensor<T>& gxLocal,
                               const LocalTensor<T>& betaLocal, 
                               const LocalTensor<T>& gammaLocal, 
                               const T alpha, 
                               const T epsilon, 
                               DeepNormTiling& tiling)
```

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/float<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/float<br>Atlas 推理系列产品AI Core，支持的数据类型为：half/float |
| isReuseSrc | 是否允许修改源操作数，默认值为false。如果开发者允许源操作数被改写，可以使能该参数，使能后能够节省部分内存空间。<br>设置为true，则本接口内部计算时复用srcLocal的内存空间，节省内存空间；设置为false，则本接口内部计算时不复用srcLocal的内存空间。<br>对于float数据类型输入支持开启该参数，half数据类型输入不支持开启该参数。<br>isReuseSrc的使用样例请参考更多样例。 |
| isBasicBlock | srcTensor的shape信息满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。基本块要求srcTensor的shape需要满足如下条件：<br>• 尾轴即H的长度为64的倍数，但不超过2040；<br>• 非尾轴长度（B*S）为8的倍数。 |

### 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| dstLocal | 输出 | 目的操作数。shape为[B, S, H]。H长度不可超过2040。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| meanLocal | 输出 | 均值，目的操作数。shape为[B, S]。meanLocal的数据类型需要与dstLocal保持一致。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| rstdLocal | 输出 | 方差，目的操作数。shape为[B, S]。rstdLocal的数据类型需要与dstLocal保持一致。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcLocal | 输入 | 源操作数，shape为[B, S, H]。srcLocal的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。H长度不可超过2040。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| gxLocal | 输入 | 源操作数，shape为[B, S, H]。gxLocal的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。H长度不可超过2040。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>该参数对应计算公式中的SubLayer(X)的计算结果。 |
| betaLocal | 输入 | 源操作数，shape为[H]。betaLocal的数据类型需要与目的操作数保持一致，长度需要32B对齐。H长度不可超过2040。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| gammaLocal | 输入 | 源操作数，shape为[H]。gammaLocal的数据类型需要与目的操作数保持一致，长度需要32B对齐。H长度不可超过2040。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 接口内部复杂计算时用于存储中间变量，由开发者提供。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>临时空间大小BufferSize的获取方式请参考15.1.5.4.12 DeepNorm Tiling。 |
| alpha | 输入 | 权重系数。数据类型需要与目的操作数一致。 |
| epsilon | 输入 | 权重系数，用来防止除零错误。数据类型需要与目的操作数一致。 |
| tiling | 输入 | DeepNorm计算所需Tiling信息，Tiling信息的获取请参考相关文档。 |
