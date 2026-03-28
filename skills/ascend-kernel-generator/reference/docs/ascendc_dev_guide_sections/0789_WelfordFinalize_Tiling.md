##### WelfordFinalize Tiling

获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

## 参数说明

### 模板参数

**表 15-771 模板参数说明**

| 参数名 | 描述 |
|--------|------|
| isReuseSource | 该参数预留，传入默认值false即可。 |

### 接口参数

**表 15-772 接口参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| outputMean | 输出 | 均值目的操作数，数据类型为float。输出的均值为1个数，需要sizeof(float)大小的空间进行保存，根据存储单元的对齐要求，开发者实际需要为outputMean分配32字节对齐的内存空间。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| outputVariance | 输出 | 方差目的操作数，数据类型为float。输出的方差为1个数，需要sizeof(float)大小的空间进行保存，根据存储单元的对齐要求，开发者实际需要为outputVariance分配32字节对齐的内存空间。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputMean | 输入 | 均值源操作数，数据类型为float。shape为[abLength]。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputVariance | 输入 | 方差源操作数，数据类型为float。shape为[abLength]。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| counts | 输入 | 源操作数，数据类型为int32_t。shape为[abLength]。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 临时空间，数据类型为uint8_t。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>接口内部复杂计算时用于存储中间变量，由开发者提供。<br>临时空间大小BufferSize的获取方式请参考WelfordFinalize Tiling。 |
| para | 输入 | 计算所需的参数信息。WelfordFinalizePara类型，定义如下：<br>`struct WelfordFinalizePara {`<br>`uint32_t rnLength;`<br>`uint32_t abLength;`<br>`uint32_t headCount;`<br>`uint32_t headCountLength;`<br>`uint32_t tailCount;`<br>`uint32_t tailCountLength;`<br>`float abRec;`<br>`float rRec;`<br>`};` |

### WelfordFinalizePara 参数说明

- **rnLength**：输入的Reduce轴，按abLength为一次计算的大小，拆分的次数。如果拆分后有尾块，则次数向上取整。
- **abLength**：Reduce轴拆分的大小。在不带counts参数的接口中，abLength=headCountLength+tailCountLength。
- **headCount**：在不带counts参数的接口中使能该参数，作为公式中非尾块的counts系数，headCount值。
- **headCountLength**：在不带counts参数的接口中使能该参数，headCount值对应的长度。
- **tailCount**：在不带counts参数的接口中使能该参数，作为公式中尾块的counts系数，tailCount值。
- **tailCountLength**：在不带counts参数的接口中使能该参数，tailCount值对应的长度。
- **abRec**：abLength的倒数，即为1/abLength的值。
- **rRec**：输入的Reduce轴拆分后，若没有尾块，表示1/(rnLength*abLength)的值，若有尾块，表示1/R的值。

## 返回值说明

无

## 约束说明

- 接口参数para.abLength的取值必须为32/sizeof(float)的整数倍。
- 接口参数para.headCountLength与para.tailCountLength的和必须等于参数para.abLength。
- 接口处理逻辑以参数para中设置的具体参数值为准，不依赖源操作数的shape信息。
- 接口参数para.tailCount为0时，禁止配置para.tailCountLength为非0值。
- 不支持源操作数与目的操作数地址重叠。
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

## 调用示例

完整的算子样例请参考welford_finalize算子样例。

```cpp
pipe.InitBuffer(sharedTmpBuffer, stackBufferSize);
AscendC::LocalTensor<uint8_t> tmpLocalTensor = sharedTmpBuffer.Get<uint8_t>();
struct AscendC::WelfordFinalizePara para = {rnLength, abLength, head, headLength, tail, tailLength, abRec, rRec};
AscendC::WelfordFinalize<false>(meanLocal, varianceLocal, inputMeanLocal, inputVarianceLocal, inputCountsLocal, tmpLocalTensor, para);
```

## WelfordFinalize Tiling

### 功能说明

Ascend C提供WelfordFinalize Tiling API，方便用户获取WelfordFinalize kernel计算时所需的Tiling参数。

获取Tiling参数主要步骤如下：

通过GetWelfordFinalizeMaxMinTmpSize获取WelfordFinalize接口计算所需最大和最小临时空间大小。

kernel侧WelfordFinalize接口的计算需要开发者预留/申请临时空间，GetWelfordFinalizeMaxMinTmpSize用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

- 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
- 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

### 函数原型

```cpp
void GetWelfordFinalizeMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

### 参数说明

**表 15-773 GetWelfordFinalizeMaxMinTmpSize 接口参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| srcShape | 输入 | 输入inputMean/inputVariance的shape信息{abLength}。 |
| typeSize | 输入 | 输入inputMean/inputVariance的数据类型大小，单位为字节。比如输入的数据类型为float，此处应传入4。 |
| isReuseSource | 输入 | 是否允许修改源操作数。该参数取值与WelfordFinalize接口一致。 |
| maxValue | 输出 | WelfordFinalize接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。<br>**说明**：maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | WelfordFinalize接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。 |

### 返回值说明

无

### 约束说明

无

### 调用示例

#### 步骤1：注册Tiling结构体

```cpp
BEGIN_TILING_DATA_DEF(WelfordFinalizeCustomTilingData) // 注册一个tiling的类，以tiling的名字作为入参
TILING_DATA_FIELD_DEF(uint32_t, isCounts); // 添加tiling字段
TILING_DATA_FIELD_DEF(uint32_t, rnLength);
TILING_DATA_FIELD_DEF(uint32_t, abLength);
TILING_DATA_FIELD_DEF(uint32_t, rLength);
TILING_DATA_FIELD_DEF(uint32_t, head);
TILING_DATA_FIELD_DEF(uint32_t, headLength);
TILING_DATA_FIELD_DEF(uint32_t, tail);
TILING_DATA_FIELD_DEF(uint32_t, tailLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(WelfordFinalizeCustom, WelfordFinalizeCustomTilingData)// 将WelfordFinalizeCustomTilingData结构体参数增加至TilingData结构体
```

#### 步骤2：Tiling实现函数

```cpp
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    WelfordFinalizeCustomTilingData tiling;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const uint32_t isCounts = *(attrs->GetAttrPointer<uint32_t>(0));
    const uint32_t rnLength = *(attrs->GetAttrPointer<uint32_t>(1));
    const uint32_t abLength = *(attrs->GetAttrPointer<uint32_t>(2));
    const uint32_t rLength = *(attrs->GetAttrPointer<uint32_t>(3));
    const uint32_t head = *(attrs->GetAttrPointer<uint32_t>(4));
    const uint32_t headLength = *(attrs->GetAttrPointer<uint32_t>(5));
    const uint32_t tail = *(attrs->GetAttrPointer<uint32_t>(6));
    const uint32_t tailLength = *(attrs->GetAttrPointer<uint32_t>(7));

    std::vector<int64_t> srcDims = {abLength};
    ge::Shape srcShape(srcDims);

    // 本样例中仅作为样例说明，通过GetWelfordFinalizeMaxMinTmpSize获取最小值并传入，来保证功能正确，
    // 开发者可以根据需要传入合适的空间大小
    uint32_t maxTmpsize = 0;
    uint32_t minTmpsize = 0;
    AscendC::GetWelfordFinalizeMaxMinTmpSize(srcShape, 4, false, maxTmpsize, minTmpsize);

    ... // 其他逻辑
    context->SetTilingKey(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

#### 步骤3：Kernel侧调用

```cpp
extern "C" __global__ __aicore__ void
welford_finalize_custom(
    GM_ADDR inputX_gm, GM_ADDR mean_gm, GM_ADDR var_gm, GM_ADDR outputMean_gm, GM_ADDR outputVariance_gm, 
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1))
    {
        if (tilingData.isCounts)
        {
            KernelWelfordFinalize<int32_t, true> op;
            op.Init(inputX_gm, mean_gm, var_gm, outputMean_gm, outputVariance_gm, tilingData.rnLength,
                    tilingData.abLength, tilingData.rLength, tilingData.head, tilingData.headLength, tilingData.tail,
                    tilingData.tailLength);
            op.Process();
        }
        else
        {
            KernelWelfordFinalize<int32_t, false> op;
            op.Init(inputX_gm, mean_gm, var_gm, outputMean_gm, outputVariance_gm, tilingData.rnLength,
                    tilingData.abLength, tilingData.rLength, tilingData.head, tilingData.headLength, tilingData.tail,
                    tilingData.tailLength);
            op.Process();
        }
    }
}
```
