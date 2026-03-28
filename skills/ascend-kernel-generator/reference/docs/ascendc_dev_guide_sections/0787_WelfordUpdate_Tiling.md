##### WelfordUpdate Tiling

获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

## 参数说明

### 模板参数说明

**表 15-768 模板参数说明**

| 参数名 | 描述 |
|--------|------|
| `T` | inputX操作数的数据类型。<br>• Atlas 推理系列产品AI Core，支持的数据类型为：half/float<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/float<br>• Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/float |
| `U` | outputMean、outputVariance、inputMean、inputVariance操作数的数据类型。<br>• Atlas 推理系列产品AI Core，支持的数据类型为：float<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：float<br>• Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：float |
| `isReuseSource` | 是否允许修改源操作数，默认值为false。如果开发者允许源操作数被改写，可以使能该参数，使能后能够节省部分内存空间。<br>• 设置为true，则本接口内部计算时复用inputX的内存空间，节省内存空间<br>• 设置为false，则本接口内部计算时不复用inputX的内存空间<br>在Atlas 推理系列产品AI Core中，该参数预留，传入默认值false即可。<br>isReuseSource的使用样例请参考更多样例。 |
| `config` | 配置非指定计算范围内的目的操作数与源操作数的复用关系。<br>WelfordUpdateConfig类型，定义如下：<br>`struct WelfordUpdateConfig {`<br>`bool isInplace = false; // 目的操作数是否复用源操作数。`<br>`};`<br>• `isInplace`：接口参数para中的abComputeLength参数指定了输入数据内层轴的计算长度，在该指定计算长度之外的输出数据具体为何值，通过本参数设置。<br>本参数表示，在指定计算长度之外的目的操作数是否复用源操作数；若复用，对于指定计算长度之外的输出，直接使用对应位置的源操作数代替输出目的操作数；若不复用，则本接口不会输出计算范围外的目的操作数。<br>– false：默认值。表示目的操作数不复用源操作数。<br>– true：表示目的操作数复用源操作数。outputMean复用inputMean，outputVariance复用inputVariance。<br>配置示例如下：<br>`constexpr WelfordUpdateConfig WFUPDATE_DEFAULT_CFG = {false};`<br>此参数一般用于配合kernel侧tiling计算的接口使用。 |

### 接口参数说明

**表 15-769 接口参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| `outputMean` | 输出 | 均值目的操作数，对应接口公式中的Meanti。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>shape和源操作数inputMean需要保持一致。 |
| `outputVariance` | 输出 | 方差中间结果目的操作数，对应接口公式中的Mi。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>shape和源操作数inputVariance需要保持一致。 |
| `inputMean` | 输入 | 均值源操作数，对应接口公式中的Meanti-1。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| `inputVariance` | 输入 | 方差中间结果源操作数，对应接口公式中的Mi-1。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| `inputX` | 输入 | 源操作数，对应接口公式中的xi。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| `sharedTmpBuffer` | 输入 | 临时空间。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>接口内部复杂计算时用于存储中间变量，由开发者提供。<br>临时空间大小BufferSize的获取方式请参考15.1.5.4.16 WelfordUpdate Tiling。 |
| `para` | 输入 | 计算所需的参数信息。WelfordUpdateParam类型，定义如下：<br>`struct WelfordUpdateParam {`<br>`uint32_t rnLength;`<br>`uint32_t abLength;`<br>`uint32_t abComputeLength;`<br>`float nRec;`<br>`};`<br>• `rnLength`：预留参数，固定设置为1<br>• `abLength`：Reduce轴拆分的大小<br>• `abComputeLength`：从输入的起始地址开始的Reduce轴实际计算长度<br>• `nRec`：取值为1/i，i为当前调用本接口的累积次数。i的取值范围为[1, n]，n为对输入数据inputX的Reduce轴切分的块数<br>各目的操作数和源操作数的shape均为[rnLength, abLength]。 |

## 返回值说明
无

## 约束说明
- 接口参数para.rnLength当前只支持取值为1
- 接口参数para.abLength的取值必须为32/sizeof(T)的整数倍
- 接口参数para.abComputeLength的取值必须大于0
- 不支持源操作数与目的操作数地址重叠
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠

## 调用示例
完整的算子样例请参考welford_update算子样例。

```cpp
#include "kernel_operator.h"

constexpr AscendC::WelfordUpdateConfig WELFORD_UPDATE_ENABLE_INPLACE_CFG = { true };
constexpr AscendC::WelfordUpdateConfig WELFORD_UPDATE_UNENABLE_INPLACE_CFG = { false };

template <typename dataType, typename dataTypeU, bool isInplace = false> class KernelWelfordUpdate {
public:
    __aicore__ inline KernelWelfordUpdate() {}
    __aicore__ inline void Init(GM_ADDR inputX_gm, GM_ADDR inputmean_gm, GM_ADDR inputvar_gm,
                                GM_ADDR outputMean_gm,
                                GM_ADDR outputVariance_gm, uint32_t nLength, uint32_t rLength, uint32_t abComputeLength)
    {
        this->nLength = nLength;
        this->rLength = rLength;
        this->abComputeLength = abComputeLength;
        totalLength = nLength * rLength;

        inputX_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputX_gm), totalLength);
        inputmean_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataTypeU *>(inputmean_gm),
                                        totalLength);
        inputvar_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataTypeU *>(inputvar_gm), totalLength);

        outputMean_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataTypeU *>(outputMean_gm),
                                         totalLength);
        outputVariance_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataTypeU *>(outputVariance_gm),
                                             totalLength);

        pipe.InitBuffer(inQueueX, 1, sizeof(dataType) * totalLength);
        pipe.InitBuffer(inQueueMean, 1, sizeof(dataTypeU) * totalLength);
        pipe.InitBuffer(inQueueVar, 1, sizeof(dataTypeU) * totalLength);
        pipe.InitBuffer(outQueueMean, 1, sizeof(dataTypeU) * totalLength);
        pipe.InitBuffer(outQueueVariance, 1, sizeof(dataTypeU) * totalLength);
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<dataType> inputXLocal = inQueueX.AllocTensor<dataType>();
        AscendC::LocalTensor<dataTypeU> inmeanLocal = inQueueMean.AllocTensor<dataTypeU>();
        AscendC::LocalTensor<dataTypeU> invarLocal = inQueueVar.AllocTensor<dataTypeU>();

        AscendC::DataCopy(inputXLocal, inputX_global, totalLength);
        AscendC::DataCopy(inmeanLocal, inputmean_global, totalLength);
        AscendC::DataCopy(invarLocal, inputvar_global, totalLength);

        inQueueX.EnQue(inputXLocal);
        inQueueMean.EnQue(inmeanLocal);
        inQueueVar.EnQue(invarLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<dataType> inputXLocal = inQueueX.DeQue<dataType>();
        AscendC::LocalTensor<dataTypeU> inmeanLocal = inQueueMean.DeQue<dataTypeU>();
        AscendC::LocalTensor<dataTypeU> invarLocal = inQueueVar.DeQue<dataTypeU>();

        AscendC::LocalTensor<dataTypeU> meanLocal = outQueueMean.AllocTensor<dataTypeU>();
        AscendC::LocalTensor<dataTypeU> varianceLocal = outQueueVariance.AllocTensor<dataTypeU>();

        struct AscendC::WelfordUpdateParam para = { nLength, rLength, abComputeLength, 0.3 };
        if constexpr (isInplace) {
            AscendC::WelfordUpdate<dataType, dataTypeU, false,
                                  WELFORD_UPDATE_ENABLE_INPLACE_CFG>(meanLocal, varianceLocal,
                                                                     inmeanLocal, invarLocal, inputXLocal, para);
        } else {
            AscendC::WelfordUpdate<dataType, dataTypeU, false,
                                  WELFORD_UPDATE_UNENABLE_INPLACE_CFG>(meanLocal, varianceLocal,
                                                                       inmeanLocal, invarLocal, inputXLocal, para);
        }

        outQueueMean.EnQue<dataTypeU>(meanLocal);
        outQueueVariance.EnQue<dataTypeU>(varianceLocal);

        inQueueX.FreeTensor(inputXLocal);
        inQueueMean.FreeTensor(inmeanLocal);
        inQueueVar.FreeTensor(invarLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dataTypeU> meanLocal = outQueueMean.DeQue<dataTypeU>();
        AscendC::LocalTensor<dataTypeU> varianceLocal = outQueueVariance.DeQue<dataTypeU>();

        AscendC::DataCopy(outputMean_global, meanLocal, totalLength);
        AscendC::DataCopy(outputVariance_global, varianceLocal, totalLength);

        outQueueMean.FreeTensor(meanLocal);
        outQueueVariance.FreeTensor(varianceLocal);
    }

private:
    AscendC::GlobalTensor<dataType> inputX_global;
    AscendC::GlobalTensor<dataTypeU> inputmean_global;
    AscendC::GlobalTensor<dataTypeU> inputvar_global;
    AscendC::GlobalTensor<dataTypeU> outputMean_global;
    AscendC::GlobalTensor<dataTypeU> outputVariance_global;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueMean;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueVar;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueMean;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueVariance;

    uint32_t nLength;
    uint32_t rLength;
    uint32_t abComputeLength;
    uint32_t totalLength;
};
```

## WelfordUpdate Tiling

### 功能说明
Ascend C提供WelfordUpdate Tiling API，方便用户获取WelfordUpdate kernel计算时所需的Tiling参数。

获取Tiling参数主要步骤如下：
具体为，通过GetWelfordUpdateMaxMinTmpSize获取WelfordUpdate接口计算所需最大和最小临时空间大小。

kernel侧WelfordUpdate接口的计算需要开发者预留/申请临时空间，GetWelfordUpdateMaxMinTmpSize用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

- 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小
- 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请

### 函数原型
```cpp
void GetWelfordUpdateMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSizeT, const uint32_t typeSizeU, const bool isReuseSource, const bool isInplace, uint32_t& maxValue, uint32_t& minValue)
```

### 参数说明

**表 15-770 GetWelfordUpdateMaxMinTmpSize 接口参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| `srcShape` | 输入 | 输入的shape信息{rnLength, abLength}。其中rnLength、abLength与WelfordUpdate接口含义一致。 |
| `typeSizeT` | 输入 | 输入(inputX)的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| `typeSizeU` | 输入 | 均值、方差(outputMean、outputVariance、inputMean、inputVariance)的数据类型大小，单位为字节。比如输入的数据类型为float，此处应传入4。 |
| `isReuseSource` | 输入 | 是否允许修改源操作数。与WelfordUpdate接口一致。 |
| `isInplace` | 输入 | 目的操作数是否复用源操作数。与WelfordUpdate接口一致。 |
| `maxValue` | 输出 | WelfordUpdate接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。<br>**说明**：maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| `minValue` | 输出 | WelfordUpdate接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。 |

### 返回值说明
无

### 约束说明
无

### 调用示例

**步骤1** 将WelfordUpdate Tiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

```cpp
BEGIN_TILING_DATA_DEF(WelfordUpdateCustomTilingData) // 注册一个tiling的类，以tiling的名字作为入参
TILING_DATA_FIELD_DEF(uint32_t, inplace); // 添加tiling字段，output是否复用input
TILING_DATA_FIELD_DEF(uint32_t, nLength);
TILING_DATA_FIELD_DEF(uint32_t, rLength);
TILING_DATA_FIELD_DEF(uint32_t, abComputeLength);
TILING_DATA_FIELD_DEF(uint32_t, nRec);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(WelfordUpdateCustom, WelfordUpdateCustomTilingData) // 将WelfordUpdateCustomTilingData结构体参数增加至TilingData结构体
```

**步骤2** Tiling实现函数中，首先调用GetWelfordUpdateMaxMinTmpSize接口获取WelfordUpdate接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入shape、剩余的可供计算的空间大小等信息获取WelfordUpdate kernel侧接口所需tiling参数。

```cpp
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    WelfordUpdateCustomTilingData tiling;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const uint32_t inplace = *(attrs->GetAttrPointer<uint32_t>(0));
    const uint32_t abComputeLength = *(attrs->GetAttrPointer<uint32_t>(1));
    const uint32_t sharedtmpbuffer = *(attrs->GetAttrPointer<uint32_t>(2));

    const gert::StorageShape *x1_shape = context->GetInputShape(1);
    const gert::Shape shape = x1_shape->GetStorageShape();
    auto nLength = shape.GetDim(0);
    auto rLength = shape.GetDim(1);

    std::vector<int64_t> srcDims = {nLength, rLength};
    ge::Shape srcShape(srcDims);

    uint32_t maxTmpsize = 0;
    uint32_t minTmpsize = 0;
    // 本样例中仅作为样例说明，通过GetWelfordUpdateMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    AscendC::GetWelfordUpdateMaxMinTmpSize(srcShape, 4, 4, false, false, maxTmpsize, minTmpsize);

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

**步骤3** 对应的kernel侧通过在核函数中调用GET_TILING_DATA获取TilingData，继而将TilingData中的WelfordUpdate Tiling信息传入WelfordUpdate接口参与计算。完整的kernel侧样例请参考15.1.5.4.15 WelfordUpdate。

```cpp
extern "C" __global__ __aicore__ void
welford_update_custom(
    GM_ADDR inputX_gm, GM_ADDR mean_gm, GM_ADDR var_gm, GM_ADDR outputMean_gm, GM_ADDR outputVariance_gm, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1))
    {
        if (tilingData.inplace)
        {
            KernelWelfordUpdate<DTYPE_INPUTX, DTYPE_U, true> op;
            op.Init(inputX_gm, mean_gm, var_gm, outputMean_gm, outputVariance_gm, tilingData.nLength, tilingData.rLength, tilingData.abComputeLength);
            op.Process();
        }
        else
        {
            KernelWelfordUpdate<DTYPE_INPUTX, DTYPE_U, false> op;
            op.Init(inputX_gm, mean_gm, var_gm, outputMean_gm, outputVariance_gm, tilingData.nLength, tilingData.rLength, tilingData.abComputeLength);
            op.Process();
        }
    }
}
```
