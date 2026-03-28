##### WelfordFinalize

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

Welford计算是一种在线计算均值和方差的方法。一方面，它可以在不存储所有样本的情况下，逐步计算所有样本的均值和方差，更适合处理海量数据；另一方面，它只需要对数据进行一次遍历，能减少访存次数，提高计算性能。本接口为Welford算法的后处理。

LayerNorm算法中Reduce轴较大的场景，可以通过切分Reduce轴，联合使用本接口与WelfordUpdate，能够实现等效计算LayerNorm。根据Reduce轴切分后是否有尾块，本接口分为如下两种计算公式：

- **不带尾块/不带counts参数场景**：
  
  其中，Mean为均值输出，Var为方差输出。Meani代表输入的第i个均值，Vari代表输入的第i个方差。Ab代表Reduce轴切分后一次计算的大小，Rn代表Reduce轴按Ab拆分的次数，代表方差系数rRec。

- **带尾块/带counts参数场景**：
  
  除上述参数含义外，countsi代表Meani对应的系数，R代表未切分的原始Reduce轴长度，代表方差系数rRec。

## 函数原型

### 通过sharedTmpBuffer入参传入临时空间

#### 不带counts参数场景
```cpp
template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalize(
    const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputMean,
    const LocalTensor<float>& inputVariance,
    const LocalTensor<uint8_t>& sharedTmpBuffer,
    WelfordFinalizePara& para
)
```

#### 带counts参数场景
```cpp
template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalize(
    const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputMean,
    const LocalTensor<float>& inputVariance,
    const LocalTensor<int32_t>& counts,
    const LocalTensor<uint8_t>& sharedTmpBuffer,
    WelfordFinalizePara& para
)
```

### 接口框架申请临时空间

#### 不带counts参数场景
```cpp
template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalize(
    const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputMean,
    const LocalTensor<float>& inputVariance,
    WelfordFinalizePara& para
)
```

#### 带counts参数场景
```cpp
template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalize(
    const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputMean,
    const LocalTensor<float>& inputVariance,
    const LocalTensor<int32_t>& counts,
    WelfordFinalizePara& para
)
```

## 临时空间说明

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持接口框架申请和开发者通过sharedTmpBuffer入参传入两种方式。

- **接口框架申请临时空间**：开发者无需申请，但是需要预留临时空间的大小。
- **通过sharedTmpBuffer入参传入**：使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过...
