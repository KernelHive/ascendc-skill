##### LayerNormGrad

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

LayerNormGrad是一个函数，用于计算LayerNorm的反向传播梯度。该接口单独使用会输出x、resForGamma；也可以和LayerNormGradBeta配合使用，输出的resForGamma传递给LayerNormGradBeta，LayerNormGradBeta接口会输出gamma和beta，配合使用时就可以同时得到x、Gamma、beta。

算法公式为：

```
pd_xl(BSH) = data_dy * data_gamma
pd_var(H) = np.sum(((-0.5) * pd_xl * (data_x - data_mean) * np.power((data_variance + EPSILON), (-1.5))), reduce_axis, keepdims=True)
pd_mean(BS1) = np.sum(((-1.0) * pd_xl * np.power((data_variance + EPSILON), (-0.5))), reduce_axis, keepdims=True) + pd_var * (1.0 / H) * np.sum(((-2.0) * (data_x - data_mean)), reduce_axis, keepdims=True)
pd_x(BSH) = pd_xl * np.power((data_variance + EPSILON), (-0.5)) + pd_var * (2.0 / H) * (data_x - data_mean) + pd_mean * (1.0 / H)
res_for_gamma(BSH) = (data_x - data_mean) * np.power((data_variance + EPSILON), (-0.5))
```

## 实现原理

以float类型，ND格式，输入为inputDy[B, S, H], inputX[B, S, H], inputVariance[B, S], inputMean[B, S], inputGamma[H]为例，描述LayerNormGrad高阶API内部算法框图，如下图所示。

图 15-56 LayerNormGrad 算法框图

计算过程分为如下几步，均在Vector上进行：

1. **ComputePdX1步骤**：计算inputDy*inputGamma，结果存储至x1Tensor；
2. **ComputePdX2步骤**：inputMean先通过Brcb将shape扩充到[B, S, H]，再计算inputX-inputMean，结果存储至x2Tensor；
3. **ComputePdVar步骤**：实现公式`np.sum(((-0.5) * x1Tensor * x2Tensor * np.power((inputVariance + EPSILON), (-1.5))))`的计算，power方法的实现通过Sqrt, Div, Mul三条基础API组合实现，结果存储至pdVarTensor；
4. **ComputePdMean步骤**：实现公式`np.sum(((-1.0) * x1Tensor * np.power((inputVariance + EPSILON), (-0.5)))) + pd_var * (1.0 / H) * np.sum(((-2.0) * (x2Tensor)))`的计算，power方法通过Sqrt, Div两条基础API组合实现，结果存储至pdMeanTensor。同时，利用中间计算结果，根据公式`x2Tensor * np.power((inputVariance + EPSILON), (-0.5))`，计算出resForGamma的结果；
5. **ComputePdX步骤**：实现公式`x1Tensor * np.power((inputVariance + EPSILON), (-0.5)) + pd_var*(2.0 / H)*(x2Tensor) + pd_mean*(1.0 / H)`的计算，结果存入outputPdX。

## 函数原型

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间大小BufferSize的获取方法：通过15.1.5.4.4 LayerNormGrad Tiling中提供的GetLayerNormGradMaxMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

临时空间支持接口框架申请和开发者通过sharedTmpBuffer入参传入两种方式，因此LayerNormGrad接口的函数原型有两种：

### 通过sharedTmpBuffer入参传入临时空间

```cpp
template <typename T, bool isReuseSource = false>
__aicore__ inline void LayerNormGrad(const LocalTensor<T> &outputPdX, const LocalTensor<T> &resForGamma, const LocalTensor<T> &inputDy, const LocalTensor<T> &inputX, const LocalTensor<T> &inputVariance, const LocalTensor<T> &inputMean, const LocalTensor<T> &inputGamma, LocalTensor<uint8_t> &sharedTmpBuffer, T epsilon, LayerNormGradTiling &tiling, const LayerNormGradShapeInfo &shapeInfo = {})
```

该方式下开发者需自行申请并管理临时内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

### 接口框架申请临时空间

```cpp
template <typename T, bool isReuseSource = false>
__aicore__ inline void LayerNormGrad(const LocalTensor<T> &outputPdX, const LocalTensor<T> &resForGamma, const LocalTensor<T> &inputDy, const LocalTensor<T> &inputX, const LocalTensor<T> &inputVariance, const LocalTensor<T> &inputMean, const LocalTensor<T> &inputGamma, T epsilon, LayerNormGradTiling &tiling, const LayerNormGradShapeInfo &shapeInfo = {})
```

该方式下开发者无需申请，但是需要预留临时空间的大小。

## 参数说明

### 表 15-747 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/float<br>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件，支持的数据类型为：half/float<br>Atlas 推理系列产品AI Core，支持的数据类型为：half/float |
| isReuseSource | 是否允许修改源操作数，默认值为false。如果开发者允许源操作数被改写，可以使能该参数，使能后能够节省部分内存空间。<br>设置为true，则本接口内部计算时复用inputX的内存空间，节省内存空间；设置为false，则本接口内部计算时不复用inputX的内存空间。<br>对于float数据类型输入支持开启该参数，half数据类型输入不支持开启该参数。<br>isReuseSource的使用样例请参考更多样例。 |

### 表 15-748 接口参数说明

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| outputPdX | 输出 | 目的操作数，shape为[B, S, H]，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。尾轴长度需要32B对齐。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| resForGamma | 输出 | 目的操作数，shape为[B, S, H]，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。尾轴长度需要32B对齐。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputDy | 输入 | 源操作数，shape为[B, S, H]，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。inputDy的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputX | 输入 | 源操作数，shape为[B, S, H]，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。inputX的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputVariance | 输入 | 方差，shape为[B, S]，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。inputVariance的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。需提前调用LayerNorm接口获取方差。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputMean | 输入 | 均值，shape为[B, S]，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。inputMean的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。需提前调用LayerNorm接口获取均值。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputGamma | 输入 | 源操作数，shape为[H]，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。inputGamma的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 共享缓冲区，用于存放API内部计算产生的临时数据。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。共享缓冲区大小的获取方式请参考 |
