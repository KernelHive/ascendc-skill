##### TopK Tiling

```markdown
## 输入参数

### topKInfo
输入 srcLocal 的 shape 信息。TopKInfo 类型，具体定义如下：

```cpp
struct TopKInfo {
    int32_t outter = 1;  // 表示输入待排序数据的外轴长度
    int32_t inner;       // 表示输入待排序数据的内轴长度，inner必须是32的整数倍
    int32_t n;           // 表示输入待排序数据的内轴的实际长度
};
```

约束说明：
- `topKInfo.inner` 必须是 32 的整数倍
- `topKInfo.inner` 是 `topKInfo.n` 进行 32 的整数倍向上补齐的值，因此 `topKInfo.n` 的大小应该满足：`1 <= topKInfo.n <= topKInfo.inner`
- Small 模式下，`topKInfo.inner` 必须设置为 32
- Normal 模式下，`topKInfo.inner` 最大值为 4096

### isLargest
输入类型为 bool：
- 取值为 `true` 时默认降序排列，获取前 k 个最大值
- 取值为 `false` 时进行升序排列，获取前 k 个最小值

## 返回值说明
无

## 约束说明

- 操作数地址偏移对齐要求请参见 15.1.2 通用说明和约束
- 不支持源操作数与目的操作数地址重叠
- 当存在 `srcLocal[i]` 与 `srcLocal[j]` 相同时，如果 `i > j`，则 `srcLocal[j]` 将首先被选出来，排在前面
- `inf` 在 Topk 中被认为是极大值
- `nan` 在 topk 中排序时无论是降序还是升序，均被排在前面
- 对于 Atlas 推理系列产品 AI Core：
  - 输入 `srcLocal` 类型是 half，模板参数 `isInitIndex` 值为 `false` 时，传入的 `topKInfo.inner` 不能大于 2048
  - 输入 `srcLocal` 类型是 half，模板参数 `isInitIndex` 值为 `true` 时，传入的 `srcIndexLocal` 中的索引值不能大于 2048

## 调用示例

本样例实现了 Normal 模式和 Small 模式的代码逻辑。算子样例工程请通过 topk 链接获取。

```cpp
if (!tmpLocal) { // 是否通过tmpLocal入参传入临时空间
    if (isSmallMode) { // Small模式
        AscendC::TopK<T, isInitIndex, isHasfinish, isReuseSrc, AscendC::TopKMode::TOPK_NSMALL>(
            dstLocalValue, dstLocalIndex, srcLocalValue, srcLocalIndex, srcLocalFinish, 
            k, topKTilingData, topKInfo, isLargest);
    } else {
        AscendC::TopK<T, isInitIndex, isHasfinish, isReuseSrc, AscendC::TopKMode::TOPK_NORMAL>(
            dstLocalValue, dstLocalIndex, srcLocalValue, srcLocalIndex, srcLocalFinish, 
            k, topKTilingData, topKInfo, isLargest);
    }
} else {
    if (tmplocalBytes % 32 != 0) {
        tmplocalBytes = (tmplocalBytes + 31) / 32 * 32;
    }
    pipe.InitBuffer(tmplocalBuf, tmplocalBytes);
    AscendC::LocalTensor<uint8_t> tmplocalTensor = tmplocalBuf.Get<uint8_t>();
    if (isSmallMode) {
        AscendC::TopK<T, isInitIndex, isHasfinish, isReuseSrc, AscendC::TopKMode::TOPK_NSMALL>(
            dstLocalValue, dstLocalIndex, srcLocalValue, srcLocalIndex, srcLocalFinish, 
            tmplocalTensor, k, topKTilingData, topKInfo, isLargest);
    } else {
        AscendC::TopK<T, isInitIndex, isHasfinish, isReuseSrc, AscendC::TopKMode::TOPK_NORMAL>(
            dstLocalValue, dstLocalIndex, srcLocalValue, srcLocalIndex, srcLocalFinish, 
            tmplocalTensor, k, topKTilingData, topKInfo, isLargest);
    }
}
```

### Normal 模式样例解析

**样例描述**：本样例为对 shape 为 (2, 32)、数据类型为 float 的矩阵进行排序的示例，分别求取每行数据的前 5 个最小值。使用 Normal 模式的接口，开发者自行传入输入数据索引，传入 `finishLocal` 来指定某些行的排序是无效排序。

**输入参数**：
- 模板参数 `T`：`float`
- 模板参数 `isInitIndex`：`true`
- 模板参数 `isHasfinish`：`true`
- 模板参数 `topkMode`：`TopKMode::TOPK_NORMAL`
- 输入数据 `finishLocal`：
  ```
  [False True False False False False False False False False False False
   False False False False False False False False False False False False
   False False False False False False False False]
  ```
  注意：DataCopy 的搬运量要求为 32byte 的倍数，因此此处 `finishLocal` 的实际有效输入是前两位 False、True，剩余的值都是进行 32bytes 向上补齐的值，并不实际参与计算。
- 输入数据 `k`：`5`
- 输入数据 `topKInfo`：
  ```cpp
  struct TopKInfo {
      int32_t outter = 2;
      int32_t inner = 32;
      int32_t n = 32;
  };
  ```
- 输入数据 `isLargest`：`false`
- 输入数据 `srcLocal`：
  ```
  [[-18096.555 -11389.83 -43112.895 -21344.77 57755.918 50911.145 24912.621 -12683.089 45088.004 -39351.043 -30153.293 11478.329 12069.15 -9215.71 45716.44 -21472.398 -37372.16 -17460.414 22498.03 21194.838 -51229.17 -51721.918 -47510.38 47899.11 43008.176 5495.8975 -24176.97 -14308.27 53950.695 7652.6035 -45169.168 -26275.518 ]
   [-9196.681 -31549.518 18589.23 -12427.927 50491.81 -20078.11 -25606.107 -34466.773 -42512.805 50584.48 35919.934 -17283.5 6488.137 -12885.134 1942.2147 -50611.96 52671.477 23179.662 25814.875 -69.73492 33906.797 -34662.61 46168.71 -52391.258 57435.332 50269.414 40935.05 21164.176 4028.458 -29022.918 -46391.133 1971.2042 ]]
  ```
- 输入数据 `srcIndexLocal`：
  ```
  [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31]
  ```

**输出数据**：
- 输出数据 `dstValueLocal` 如下，每行长度是 `k_pad`，其中每条数据的前 5 个值就是该条的前 5 个最小值。后面的三个值是随机值：
  ```
  [[-51721.918 -51229.17 -47510.38 -45169.168 -43112.895 0. 0. 0. ]
   [-52391.258 -50611.96 -46391.133 -42512.805 -34662.61 0. 0. 0. ]]
  ```
- 输出数据 `dstIndexLocal` 如下：
  - 每行长度是 `kpad_index`，其中每条数据的前 5 个值就是该条的前 5 个最小值对应的索引。后面的三个值是随机值
  - 由于第二行数据对应的 `finishLocal` 为 true，说明第二行数据的排序是无效的，所以其输出的索引值均为内轴实际长度 32
  ```
  [[21 20 22 30 2 0 0 0]
   [32 32 32 32 32 0 0 0]]
  ```

### Small 模式样例解析

**样例描述**：本样例为对 shape 为 (4, 17)、类型为 float 的输入数据进行排序的示例，求取每行数据的前 8 个最大值。使用 Small 模式的接口，开发者自行传入输入数据索引。

**输入参数**：
- 模板参数 `T`：`float`
- 模板参数 `isInitIndex`：`true`
- 模板参数 `isHasfinish`：`false`
- 模板参数 `topkMode`：`TopKMode::TOPK_NSMALL`
- 输入数据 `finishLocal`：`LocalTensor<bool> finishLocal`，不需要赋值
- 输入数据 `k`：`8`
- 输入数据 `topKInfo`：
  ```cpp
  struct TopKInfo {
      int32_t outter = 4;
      int32_t inner = 32;
      int32_t n = 17;
  };
  ```
- 输入数据 `isLargest`：`true`
- 输入数据 `srcLocal`：此处 `n=17`，不是 32 的整数倍时，将其向上补齐到 32，填充内容为 `-inf`
  ```
  [[ 55492.18 27748.229 -51100.11 19276.926 14828.149 -20771.824 57553.4 -21504.092 -57423.414 142.36443 -5223.254 54669.473 54519.184 10165.924 -658.4564 2264.2397 -52942.883 -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf]
   [-52849.074 57778.72 37069.496 16273.109 -25150.637 -35680.5 -15823.097 4327.308 -35853.86 -7052.2627 44148.117 -17515.457 -18926.059 -1650.6737 21753.582 -2589.2822 39390.4 -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf]
   [-17539.186 -15220.923 29945.332 -4088.1514 28482.525 29750.484 -46082.03 31141.16 23140.047 8461.174 39955.844 29401.35 53757.543 33584.566 -3543.6284 -38318.344 22212.41 -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf]
   [ -9970.768 -9191.963 -17903.045 2211.4912 47037.562 -41114.824 13305.985 59926.07 -24316.797 -6462.8896 5699.733 -5873.5015 15695.861 -38492.004 19581.654 -36877.68 27090.158 -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf]]
  ```
- 输入数据 `srcIndexLocal`：
  ```
  [[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31]
   [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31]
   [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31]
   [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31]]
  ```

**输出数据**：
- 输出数据 `dstValueLocal`：输出每行数据的前 8 个最大值
  ```
  [[57553.4 55492.18 54669.473 54519.184 27748.229 19276.926 14828.149 10165.924]
   [57778.72 44148.117 39390.4 37069.496 21753.582 16273.109 4327.308 -1650.6737]
   [53757.543 39955.844 33584.566 31141.16 29945.332 29750.484 29401.35 28482.525]
   [59926.07 47037.562 27090.158 19581.654 15695.861 13305.985 5699.733 2211.4912]]
  ```
- 输出数据 `dstIndexLocal`：输出每行数据的前 8 个最大值索引
  ```
  [[6 0 11 12 1 3 4 13]
   [1 10 16 2 14 3 7 13]
   [12 10 13 7 2 5 11 4]
   [7 4 16 14 12 6 10 3]]
  ```

## TopK Tiling

### 功能说明

用于获取 Topk Tiling 参数。

Ascend C 提供 Topk Tiling API，方便用户获取 Topk kernel 计算时所需的 Tiling 参数。阅读本节之前，请先参考 Tiling 实现了解 Tiling 实现基本流程。

获取 Tiling 参数主要分为如下两步：
1. 获取 Topk 接口计算所需最小和最大临时空间大小，注意该步骤不是必须的，只是作为一个参考，供合理分配计算空间
2. 获取 Topk kernel 侧接口所需 tiling 参数

Topk Tiling 结构体的定义如下，开发者无需关注该 tiling 结构的具体信息，只需要传递到 kernel 侧，传入 Topk 高阶 API 接口，直接进行使用即可。

```cpp
struct TopkTiling {
    int32_t tmpLocalSize = 0;
    int32_t allDataSize = 0;
    int32_t innerDataSize = 0;
    uint32_t sortRepeat = 0;
    int32_t mrgSortRepeat = 0;
    int32_t kAlignFourBytes = 0;
    int32_t kAlignTwoBytes = 0;
    int32_t maskOffset = 0;
    int32_t maskVreducev2FourBytes = 0;
    int32_t maskVreducev2TwoBytes = 0;
    int32_t mrgSortSrc1offset = 0;
    int32_t mrgSortSrc2offset = 0;
    int32_t mrgSortSrc3offset = 0;
    int32_t mrgSortTwoQueueSrc1Offset = 0;
    int32_t mrgFourQueueTailPara1 = 0;
    int32_t mrgFourQueueTailPara2 = 0;
    int32_t srcIndexOffset = 0;
    uint32_t copyUbToUbBlockCount = 0;
    int32_t topkMrgSrc1MaskSizeOffset = 0;
    int32_t topkNSmallSrcIndexOffset = 0;
    uint32_t vreduceValMask0 = 0;
    uint32_t vreduceValMask1 = 0;
    uint32_t vreduceIdxMask0 = 0;
    uint32_t vreduceIdxMask1 = 0;
    uint16_t vreducehalfValMask0 = 0;
    uint16_t vreducehalfValMask1 = 0;
    uint16_t vreducehalfValMask2 = 0;
    uint16_t vreducehalfValMask3 = 0;
    uint16_t vreducehalfValMask4 = 0;
    uint16_t vreducehalfValMask5 = 0;
    uint16_t vreducehalfValMask6 = 0;
    uint16_t vreducehalfValMask7 = 0;
};
```

### 函数原型

```cpp
bool GetTopKMaxMinTmpSize(const platform_ascendc::PlatformAscendC &ascendcPlatform, 
                         const int32_t inner, 
                         const int32_t outter, 
                         const bool isReuseSource, 
                         const bool isInitIndex, 
                         enum TopKMode mode, 
                         const bool isLargest, 
                         const uint32_t dataTypeSize, 
                         uint32_t &maxValue, 
                         uint32_t &minValue)

bool TopKTilingFunc(const platform_ascendc::PlatformAscendC &ascendcPlatform, 
                   const int32_t inner, 
                   const int32_t outter, 
                   const int32_t k, 
                   const uint32_t dataTypeSize, 
                   const bool isInitIndex, 
                   enum TopKMode mode, 
                   const bool isLargest, 
                   optiling::TopkTiling &topKTiling)
```

### 参数说明

#### GetTopKMaxMinTmpSize 接口参数列表

| 参数 | 输入/输出 | 功能描述 |
|------|-----------|----------|
| ascendcPlatform | 输入 | 传入硬件平台的信息，PlatformAscendC 定义请参见构造及析构函数 |
| inner | 输入 | 表示 TopK 接口输入 srcLocal 的内轴长度，该参数的取值为 32 的整数倍 |
| outter | 输入 | 表示 TopK 接口输入 srcLocal 的外轴长度 |
| isReuseSource | 输入 | 中间变量是否能够复用输入内存。与 kernel 侧接口的 isReuseSrc 保持一致 |
| isInitIndex | 输入 | 是否传入输入数据对应的索引，与 kernel 侧接口一致 |
| mode | 输入 | 选择 TopKMode::TOPK_NORMAL 模式或者 TopKMode::TOPK_NSMALL 模式，与 kernel 侧接口一致 |
| isLargest | 输入 | 表示降序/升序，true 表示降序，false 表示升序。与 kernel 侧接口一致 |
| dataTypeSize | 输入 | 参与计算的 srcLocal 数据类型的大小，比如 half=2，float=4 |
| maxValue | 输出 | Topk 接口内部完成计算需要的最大临时空间大小，单位是 Byte。<br>说明：maxValue 仅作为参考值，有可能大于 Unified Buffer 剩余空间的大小，该场景下，开发者需要根据 Unified Buffer 剩余空间的大小来选取合适的临时空间大小 |
| minValue | 输出 | Topk 接口内部完成计算需要的最小临时空间大小，单位是 Byte |

#### TopKTilingFunc 接口参数列表

| 参数 | 输入/输出 | 功能描述 |
|------|-----------|----------|
| ascendcPlatform | 输入 | 传入硬件平台的信息，PlatformAscendC 定义请参见构造及析构函数 |
| inner | 输入 | 表示 TopK 接口输入 srcLocal 的内轴长度，该参数的取值为 32 的整数倍 |
| outter | 输入 | 表示 TopK 接口输入 srcLocal 的外轴长度 |
| k | 输入 | 获取前 k 个最大值或最小值及其对应的索引 |
| dataTypeSize | 输入 | 参与计算的 srcLocal 数据类型的大小，比如 half=2，float=4 |
| isInitIndex | 输入 | 是否传入输入数据对应的索引，与 kernel 侧接口一致 |
| mode | 输入 | 选择 TopKMode::TOPK_NORMAL 模式或者 TopKMode::TOPK_NSMALL 模式，与 kernel 侧接口一致 |
| isLargest | 输入 | 表示降序/升序，true 表示降序，false 表示升序。与 kernel 侧接口一致 |
| topKTiling | 输出 | 输出 Topk 接口所需的 tiling 信息 |

### 返回值说明

- `GetTopKMaxMinTmpSize` 返回值为 `true/false`，`true` 表示成功拿到 Topk 接口内部计算需要的最大和最小临时空间大小；`false` 表示获取失败
- `TopKTilingFunc` 返回值为 `true/false`，`true` 表示成功拿到 Topk 的 Tiling 各项参数值；`false` 表示获取失败

### 约束说明
无

### 调用示例

如下样例介绍了使用 Topk 高阶 API 时 host 侧获取 Tiling 参数的流程以及该参数如何在 kernel 侧使用。

#### 步骤 1
将 Topk Tiling 结构体参数增加至 TilingData 结构体，作为 TilingData 结构体的一个字段。

```cpp
namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tilenum);
    //添加其他tiling字段
    ...
    TILING_DATA_FIELD_DEF(int32_t, k);
    TILING_DATA_FIELD_DEF(bool, islargest);
    TILING_DATA_FIELD_DEF(bool, isinitindex);
    TILING_DATA_FIELD_DEF(bool, ishasfinish);
    TILING_DATA_FIELD_DEF(uint32_t, tmpsize);
    TILING_DATA_FIELD_DEF(int32_t, outter);
    TILING_DATA_FIELD_DEF(int32_t, inner);
    TILING_DATA_FIELD_DEF(int32_t, n);
    TILING_DATA_FIELD_DEF(int32_t, order);
    TILING_DATA_FIELD_DEF(int32_t, sorted);
    TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TopkCustom, TilingData)
}
```

#### 步骤 2
Tiling 实现函数中，首先调用 `GetTopKMaxMinTmpSize` 接口获取 Topk 接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小；然后根据输入 shape 等信息获取 Topk kernel 侧接口所需 tiling 参数。

```cpp
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
const int32_t OUTTER = 2;
const int32_t INNER = 32;
const int32_t N = 32;
const int32_t K = 8;
const bool IS_LARGEST = true;
const bool IS_INITINDEX = true;
const bool IS_REUSESOURCE = false;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.set_k(K);
    tiling.set_outter(OUTTER);
    tiling.set_inner(INNER);
    tiling.set_n(N);
    tiling.set_islargest(IS_LARGEST);
    tiling.set_isinitindex(IS_INITINDEX);
    // 设置其他Tiling参数
    ...
    
    // 本样例中仅作为样例说明，通过GetTopKMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小。
    uint32_t maxsize = 0;
    uint32_t minsize = 0;
    uint32_t dtypesize = 4; // float类型
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    AscendC::TopKTilingFunc(ascendcPlatform, tiling.inner, tiling.outter, tiling.k, dtypesize, tiling.isinitindex, 
                           AscendC::TopKMode::TOPK_NSMALL, tiling.islargest, tiling.topkTilingData);
    AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, tiling.inner, tiling.outter, IS_REUSESOURCE, 
                                 tiling.isinitindex, AscendC::TopKMode::TOPK_NSMALL, tiling.islargest, 
                                 dtypesize, maxsize, minsize);
    tiling.set_tmpsize(minsize);
    ... // 其他逻辑
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

#### 步骤 3
对应的 kernel 侧通过在核函数中调用 `GET_TILING_DATA` 获取 TilingData，继而将 TilingData 中的 Topk Tiling 信息传入 Topk 接口参与计算。完整的 kernel 侧样例请参考调用示例。

```cpp
extern "C" __global__ __aicore__ void topk_custom(GM_ADDR srcVal, GM_ADDR srcIdx, GM_ADDR finishLocal, 
                                                 GM_ADDR dstVal, GM_ADDR dstIdx, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    
    KernelTopK<float, true, true, false, false, AscendC::TopKMode::TOPK_NSMALL> op;
    op.Init(srcVal, srcIdx, finishLocal, dstVal, dstIdx, tilingData.k, tilingData.islargest, tilingData.tmpsize,
           tilingData.outter, tilingData.inner, tilingData.n, tilingData.topkTilingData);
    op.Process();
}
```
```
