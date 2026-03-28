##### WelfordUpdate

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

Welford是一种在线计算均值和方差的方法。一方面，它可以在不存储所有样本的情况下，逐步计算所有样本的均值和方差，更适合处理海量数据；另一方面，它只需要对数据进行一次遍历，能减少访存次数，提高计算性能。本接口为Welford算法的预处理。

LayerNorm算法中Reduce轴较大的场景，可以通过切分Reduce轴，联合使用本接口与WelfordFinalize，实现等效计算LayerNorm。

如下图所示，切分数据的Reduce轴，假设切分后每块数据的形状为[1, k]，每块数据标号为1，2，3，…，n。

**图 15-57 Reduce 轴切分示意图**

本接口的计算公式如下。进行上述的数据切分后，分n次调用本接口，切分后的每块数据均完成如下公式的计算。

上式中，xi、Meanti、Mi的形状均为[1, k]，xi表示切分后的第i块数据，Meanti表示第i次调用本接口得到的前i块数据的均值，Mi表示第i次调用本接口得到的前i块数据的方差中间结果（即为求方差而保存的中间计算结果，本节后续内容中写作方差中间结果）。其中，第一次调用本接口，即i=1时，公式中的Meant0和M0由用户定义为形状[1, k]、取值全0的数据。

Meantn的计算过程示意如下图，调用n次本接口后，得到形状为[1, k]的Meantn和Mn，Meantn和Mn用于后续WelfordFinalize接口的计算。

**图 15-58 均值 Meantn 计算过程示意图**

## 函数原型

### 通过sharedTmpBuffer入参传入临时空间

```cpp
template <typename T, typename U, bool isReuseSource = false, const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
__aicore__ inline void WelfordUpdate(
    const LocalTensor<U>& outputMean,
    const LocalTensor<U>& outputVariance,
    const LocalTensor<U>& inputMean,
    const LocalTensor<U>& inputVariance,
    const LocalTensor<T>& inputX,
    const LocalTensor<uint8_t>& sharedTmpBuffer,
    const WelfordUpdateParam& para
)
```

### 接口框架申请临时空间

```cpp
template <typename T, typename U, bool isReuseSource = false, const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
__aicore__ inline void WelfordUpdate(
    const LocalTensor<U>& outputMean,
    const LocalTensor<U>& outputVariance,
    const LocalTensor<U>& inputMean,
    const LocalTensor<U>& inputVariance,
    const LocalTensor<T>& inputX,
    const WelfordUpdateParam& para
)
```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持接口框架申请和开发者通过sharedTmpBuffer入参传入两种方式。

- **接口框架申请临时空间**：开发者无需申请，但是需要预留临时空间的大小。
- **通过sharedTmpBuffer入参传入**：使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过
