##### LayerNormGradBeta

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

LayerNormGradBeta接口用于获取反向beta/gamma的数值，和LayerNormGrad共同输出pdx、gamma和beta。

算法公式为：

## 函数原型

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间大小BufferSize的获取方法：通过15.1.5.4.6 LayerNormGradBeta Tiling中提供的GetLayerNormGradBetaMaxMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

临时空间支持接口框架申请和开发者通过sharedTmpBuffer入参传入两种方式，因此LayerNormGradBeta接口的函数原型有两种：

- **通过sharedTmpBuffer入参传入临时空间**
  ```cpp
  template <typename T, bool isReuseSource = false>
  __aicore__ inline void LayerNormGradBeta(const LocalTensor<T> &outputPdGamma, 
                                          const LocalTensor<T> &outputPdBeta, 
                                          const LocalTensor<T> &resForGamma, 
                                          const LocalTensor<T> &inputDy, 
                                          const LocalTensor<uint8_t> &sharedTmpBuffer, 
                                          const LayerNormGradBetaTiling &tiling)
  ```
  该方式下开发者需自行申请并管理临时内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

- **接口框架申请临时空间**
  ```cpp
  template <typename T, bool isReuseSource = false>
  __aicore__ inline void LayerNormGradBeta(const LocalTensor<T> &outputPdGamma, 
                                          const LocalTensor<T> &outputPdBeta, 
                                          const LocalTensor<T> &resForGamma, 
                                          const LocalTensor<T> &inputDy, 
                                          LayerNormGradBetaTiling &tiling)
  ```
  该方式下开发者无需申请，但是需要预留临时空间的大小。

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/float<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/float<br>Atlas 推理系列产品AI Core，支持的数据类型为：half/float |
| isReuseSource | 是否允许修改源操作数，默认值为false。如果开发者允许源操作数被改写，可以使能该参数，使能后能够节省部分内存空间。<br>设置为true，则本接口内部计算时复用inputDy的内存空间，节省内存空间；设置为false，则本接口内部计算时不复用inputDy的内存空间。<br>对于float数据类型输入支持开启该参数，half数据类型输入不支持开启该参数。<br>isReuseSource的使用样例请参考更多样例。 |

### 接口参数说明

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| outputPdGamma | 输出 | 目的操作数，shape为[H]，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。尾轴长度需要32B对齐<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| outputPdBeta | 输出 | 目的操作数，shape为[H]，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。尾轴长度需要32B对齐<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| resForGamma | 输入 | 源操作数，shape为[B, S, H]，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。<br>resForGamma的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。需提前调用LayerNormGrad接口获取resForGamma参数值。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputDy | 输入 | 源操作数，shape为[B, S, H]，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。inputDy的数据类型需要与目的操作数保持一致，尾轴长度需要32B对齐。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 共享缓冲区，用于存放API内部计算产生的临时数据。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。共享缓冲区大小的获取方式请参考 |
