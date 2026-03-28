###### SoftMax

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 AI Core | √ |
| Atlas 推理系列产品 Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

将输入 tensor `[m0, m1, ...mt, n]`（t 大于等于 0）的非尾轴长度相乘的结果看作 `m`，则输入 tensor 的 shape 看作 `[m, n]`。对输入 tensor `[m, n]` 按行做 SoftMax 计算。

为方便理解，通过 Python 脚本实现的方式，表达其计算公式（以输入为 ND 格式为例）如下，其中 `src` 是源操作数（输入），`dst`、`sum`、`max` 为目的操作数（输出）。

```python
def softmax(src):
    # 基于 last 轴进行 rowmax（按行取最大值）处理
    max = np.max(src, axis=-1, keepdims=True)
    sub = src - max
    exp = np.exp(sub)
    # 基于 last 轴进行 rowsum（按行求和）处理
    sum = np.sum(exp, axis=-1, keepdims=True)
    dst = exp / sum
    return dst, max, sum
```

当输入的数据排布格式不同时，内部的 reduce 过程会有所不同：

- 当输入为 ND 格式时，内部的 reduce 过程按 last 轴进行
- 当输入为 NZ 格式时，内部的 reduce 过程按照 last 轴和 first 轴进行

图 15-46 ND 格式的 reduce 过程

图 15-47 NZ 格式的 reduce 过程

## 实现原理

以 float 类型，ND 格式，shape 为 `[m, k]` 的输入 Tensor 为例，描述 SoftMax 高阶 API 内部算法框图，如下图所示。

图 15-48 SoftMax 算法框图

计算过程分为如下几步，均在 Vector 上进行：

1. **reducemax 步骤**：对输入 x 的每一行数据求最大值得到 `[m, 1]` 的结果，计算结果会保存到一个临时空间 temp 中
2. **broadcast 步骤**：对 temp 中的数据 `[m, 1]` 做一个按 datablock 为单位的填充，比如 float 类型下，把 `[m, 1]` 扩展成 `[m, 8]`，同时输出 max
3. **sub 步骤**：对输入 x 的所有数据按行减去 max
4. **exp 步骤**：对 sub 之后的所有数据求 exp
5. **reducesum 步骤**：对 exp 后的结果的每一行数据求和得到 `[m, 1]`，计算结果会保存到临时空间 temp 中
6. **broadcast 步骤**：对 temp(`[m, 1]`) 做一个按 datablock 为单位的填充，比如 float 类型下，把 `[m, 1]` 扩展成 `[m, 8]`，同时输出 sum
7. **div 步骤**：对 exp 后的结果的所有数据按行除以 sum，得到最终结果

## 函数原型

### 接口框架申请临时空间

#### LocalTensor 的数据类型相同

```cpp
template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
```

#### LocalTensor 的数据类型不同

```cpp
template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMax(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
```

#### 不带 sumTensor 和 maxTensor 参数

```cpp
template <typename T, bool isReuseSource = false, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
```

### 通过 sharedTmpBuffer 入参传入临时空间

#### LocalTensor 的数据类型相同

```cpp
template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
```

#### LocalTensor 的数据类型不同

```cpp
template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMax(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
```

#### 不带 sumTensor 和 maxTensor 参数

```cpp
template <typename T, bool isReuseSource = false, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持接口框架申请和开发者通过 sharedTmpBuffer 入参传入两种方式。

- **接口框架申请临时空间**：开发者无需申请，但是需要预留临时空间的大小
- **通过 sharedTmpBuffer 入参传入**：使用该 tensor 作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理 sharedTmpBuffer 内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。具体内存复用方式可参考《Ascend C 最佳实践》中的性能优化 > 内存优化 > 算子与高阶 API 共享临时 Buffer 章节

接口框架申请的方式，开发者需要预留临时空间；通过 sharedTmpBuffer 传入的情况，开发者需要为 tensor 申请空间。临时空间大小 BufferSize 的获取方式如下：通过 SoftMax/SimpleSoftMax Tiling 中提供的 GetSoftMaxMaxTmpSize/GetSoftMaxMinTmpSize 接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

## 参数说明

### 表 15-683 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/float<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/float<br>Atlas 200I/500 A2 推理产品，支持的数据类型为：half/float<br>Atlas 推理系列产品 AI Core，支持的数据类型为：half/float |
| isReuseSource | 该参数预留，传入默认值 false 即可 |
| isBasicBlock | srcTensor 和 dstTensor 的 shape 信息和 Tiling 切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。是否满足基本块的要求，可以采用如下两种方式之一判断：<br>• srcTensor 和 dstTensor 的 shape 信息 `[m,n]` 需要满足如下条件：<br>  - 尾轴长度 n 小于 2048 并且大于等于 256/sizeof(T)（即 half 场景下 n 最小为 128，float 场景下 n 最小为 64），同时 n 是 64 的倍数<br>  - 非尾轴长度的乘积 m 为 8 的倍数<br>• 在 Tiling 实现中，通过调用 IsBasicBlockInSoftMax 判断 Tiling 切分策略是否满足基本块的切分要求<br>针对 Atlas 200I/500 A2 推理产品，该参数为预留参数，暂未启用，为后续的功能扩展做保留，保持默认值即可 |
| isDataFormatNZ | 当前输入输出的数据格式是否为 NZ 格式，默认数据格式为 ND，即默认取值为 false<br>针对 Atlas 200I/500 A2 推理产品，不支持配置为 NZ 格式 |
| config | 结构体模板参数，此参数可选配，SoftmaxConfig 类型，具体定义如下：<br>`struct SoftmaxConfig{`<br>`bool isCheckTiling = true; // 是否需要检查 shape 和 tiling 的一致性；若不一致，API 内会根据 shape 重新计算所需 tiling。默认取值 true：API 内部会检查一致性`<br>`uint32_t oriSrcM = 0; // 原始非尾轴长度的乘积。设置该参数后，将 shape 常量化，编译过程中使用常量化的 shape`<br>`uint32_t oriSrcK = 0; // 原始尾轴长度。设置该参数后，将 shape 常量化，编译过程中使用常量化的 shape`<br>`};`<br>配置示例如下：<br>`constexpr SoftmaxConfig SOFTMAX_DEFAULT_CFG = {true, 0, 0};`<br>此参数一般用于配合 kernel 侧 tiling 计算的接口使用<br>注意：设置了 oriSrcM 与 oriSrcK 后，模板参数 isBasicBlock 不生效，计算数据是否为基本块由 API 内部判断并处理<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持该参数<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持该参数<br>针对 Atlas 推理系列产品 AI Core，该参数为预留参数，暂未启用，保持默认值即可<br>针对 Atlas 200I/500 A2 推理产品，该参数为预留参数，暂未启用，保持默认值即可 |

### 表 15-684 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| dstTensor | 输出 | 目的操作数<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT<br>dst 的 shape 和源操作数 src 一致 |
| sumTensor | 输出 | 目的操作数<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT<br>用于保存 SoftMax 计算过程中 reducesum 的结果<br>• sumTensor 的 last 轴长度固定为 32Byte，即一个 datablock 长度。该 datablock 中的所有数据为同一个值，比如 half 数据类型下，该 datablock 中的 16 个数均为相同的 reducesum 的值<br>• 非 last 轴的长度与 dst 保持一致 |
| maxTensor | 输出 | 目的操作数<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT<br>用于保存 SoftMax 计算过程中 reducemax 的结果<br>• maxTensor 的 last 轴长度固定为 32Byte，即一个 datablock 长度。该 datablock 中的所有数据为同一个值，比如 half 数据类型下，该 datablock 中的 16 个数均为相同的 reducemax 的值<br>• 非 last 轴的长度与 dst 保持一致 |
| srcTensor | 输入 | 源操作数<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT<br>last 轴长度需要 32Byte 对齐 |
| sharedTmpBuffer | 输入 | 临时空间<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT<br>接口内部复杂计算时用于存储中间变量，由开发者提供<br>临时空间大小 BufferSize 的获取方式请参考 SoftMax/SimpleSoftMax Tiling |
| tiling | 输入 | SoftMax 计算所需 Tiling 信息，Tiling 信息的获取请参考 SoftMax/SimpleSoftMax Tiling |
| softmaxShapeInfo | 输入 | src 的 shape 信息。SoftMaxShapeInfo 类型，具体定义如下：<br>`struct SoftMaxShapeInfo {`<br>`uint32_t srcM; // 非尾轴长度的乘积`<br>`uint32_t srcK; // 尾轴长度，必须 32Byte 对齐`<br>`uint32_t oriSrcM; // 原始非尾轴长度的乘积`<br>`uint32_t oriSrcK; // 原始尾轴长度`<br>`};`<br>需要注意，当输入输出的数据格式为 NZ 格式时，尾轴长度为 reduce 轴长度即图 15-47 中的 W0*W1，非尾轴为 H0*H1 |

## 返回值说明

无

## 约束说明

- src 和 dst 的 Tensor 空间可以复用
- sumTensor 和 maxTensor 为输出，并且 last 轴长度必须固定 32Byte，非 last 轴大小需要和 src 以及 dst 保持一致
- sumTensor 和 maxTensor 的数据类型需要保持一致
- 操作数地址对齐要求请参见通用地址对齐约束
- 不支持 sharedTmpBuffer 与源操作数和目的操作数地址重叠
- 当参数 softmaxShapeInfo 中 `srcM != oriSrcM` 或者 `srcK != oriSrcK` 时，开发者需要对 GM 上的原始输入 `(oriSrcM, oriSrcK)` 在 M 或 K 方向补齐数据到 `(srcM, srcK)`，补齐的数据会参与部分运算，在输入输出复用的场景下，API 的计算结果会覆盖 srcTensor 中补齐的原始数据，在输入输出不复用的场景下，API 的计算结果会覆盖 dstTensor 中对应 srcTensor 补齐位置的数据

## 调用示例

本样例中输入 src 和输出 dst 的 shape 大小为 `[320,64]`，中间计算结果 sumTensor 和 maxTensor 的 shape 大小为 `[320,16]`，数据类型均为 half，输入输出的数据排布格式为 ND，src 和 dst 空间不复用，不使能基本块。更多算子样例请参考 softmax 算子样例。

```cpp
AscendC::LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
AscendC::LocalTensor<T> sumTempLocal = sumQueue.AllocTensor<T>();
AscendC::LocalTensor<T> maxTempLocal = maxQueue.AllocTensor<T>();
AscendC::LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

AscendC::SoftMaxShapeInfo srcShape = {height, width, height, width};
AscendC::SoftMax<T>(dstLocal, sumTempLocal, maxTempLocal, srcLocal, tiling, srcShape);

// AscendC::SoftMax<T, false, false, false, static_config>(dstLocal, sumTempLocal,
// maxTempLocal, srcLocal, tiling, srcShape); 使用 SoftmaxConfig 类型的参数 static_config，传入模板参数将 shape 常量化

outQueueDst.EnQue<T>(dstLocal);
maxQueue.FreeTensor(maxTempLocal);
sumQueue.FreeTensor(sumTempLocal);
inQueueSrc.FreeTensor(srcLocal);
```
