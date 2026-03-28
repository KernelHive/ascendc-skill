##### Normalize

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

LayerNorm中，已知均值和方差，计算shape为[A，R]的输入数据的标准差的倒数rstd和y，其计算公式如下：

```
rstd = 1 / sqrt(Var + ε)
y = (x - E) * rstd * γ + β
```

其中：
- E和Var分别代表输入在R轴的均值、方差
- γ为缩放系数
- β为平移系数
- ε为防除零的权重系数

## 函数原型

### 通过sharedTmpBuffer入参传入临时空间

```cpp
template <typename U, typename T, bool isReuseSource = false, const NormalizeConfig& config = NLCFG_NORM>
__aicore__ inline void Normalize(
    const LocalTensor<T>& output,
    const LocalTensor<float>& outputRstd,
    const LocalTensor<float>& inputMean,
    const LocalTensor<float>& inputVariance,
    const LocalTensor<T>& inputX,
    const LocalTensor<U>& gamma,
    const LocalTensor<U>& beta,
    const LocalTensor<uint8_t>& sharedTmpBuffer,
    const float epsilon,
    const NormalizePara& para
)
```

### 接口框架申请临时空间

```cpp
template <typename U, typename T, bool isReuseSource = false, const NormalizeConfig& config = NLCFG_NORM>
__aicore__ inline void Normalize(
    const LocalTensor<T>& output,
    const LocalTensor<float>& outputRstd,
    const LocalTensor<float>& inputMean,
    const LocalTensor<float>& inputVariance,
    const LocalTensor<T>& inputX,
    const LocalTensor<U>& gamma,
    const LocalTensor<U>& beta,
    const float epsilon,
    const NormalizePara& para
)
```

## 临时空间说明

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持两种方式：

- **接口框架申请临时空间**：开发者无需申请，但是需要预留临时空间的大小
- **通过sharedTmpBuffer入参传入**：使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过...
