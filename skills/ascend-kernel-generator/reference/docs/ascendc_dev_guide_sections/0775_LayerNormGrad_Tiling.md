##### LayerNormGrad Tiling

```markdown
### 输入参数说明

- **类型**：LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT
- **epsilon**：输入防除零的权重系数
- **tiling**：输入LayerNormGrad计算所需Tiling信息
- **shapeInfo**：输入表示LayerNormGrad各个输入的数据排布格式Format
  - 默认值表示输入的Format为ND
  - 支持的取值为DataFormat::ND
  - LayerNormGradShapeInfo类型定义如下：

```cpp
struct LayerNormGradShapeInfo {
    DataFormat dataFormat = DataFormat::ND;
};
```

### 返回值说明
无

### 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束
- 源操作数和目的操作数的Tensor空间可以复用
- 仅支持输入shape为ND格式
- 输入数据不满足对齐要求时，开发者需要进行补齐，补齐的数据应设置为0，防止出现异常值从而影响网络计算
- 不支持对尾轴H轴的切分

### 调用示例

本样例中：
- 输入inputX和inputDy的shape为[2, 32, 16]
- inputVariance和inputMean的shape为[2, 32]
- inputGamma的shape为[16]
- 输出outputPdX和resForGamma的shape为[2, 32, 16]
- 数据排布均为ND格式，数据类型均为float
- 不复用源操作数的内存空间

完整调用样例请参考layernorm_grad：

```cpp
#include "kernel_operator.h"

namespace MyCustomKernel {
struct VecTiling {
    LayerNormGradTiling layernormGradTilingData;
    float epsilon = 0;
};

template <bool isReuseSource = false> 
class KernelLayernormGrad {
public:
    __aicore__ inline KernelLayernormGrad() {}
    
    __aicore__ inline void Init(GM_ADDR inputXGm, GM_ADDR inputDyGm, GM_ADDR inputVarianceGm,
                               GM_ADDR inputMeanGm, GM_ADDR inputGammaGm, GM_ADDR outputPdXGm, 
                               GM_ADDR resForGammaGm, VecTiling tilingData)
    {
        this->epsilon = tilingData.epsilon;
        tiling_ = tilingData.layernormGradTilingData;
        this->bLength = tiling_.bLength;
        this->sLength = tiling_.sLength;
        this->hLength = tiling_.hLength;
        bshLength = bLength * sLength * hLength;
        bsLength = bLength * sLength;
        
        inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputXGm), bshLength);
        inputDyGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputDyGm), bshLength);
        inputVarianceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputVarianceGm), bsLength);
        inputMeanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputMeanGm), bsLength);
        inputGammaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputGammaGm), hLength);
        outputPdXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputPdXGm), bshLength);
        outputResForGammaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(resForGammaGm), bshLength);
        
        pipe.InitBuffer(inQueueX, 1, sizeof(float) * bshLength);
        pipe.InitBuffer(inQueueDy, 1, sizeof(float) * bshLength);
        pipe.InitBuffer(inQueueVariance, 1, sizeof(float) * bsLength);
        pipe.InitBuffer(inQueueMean, 1, sizeof(float) * bsLength);
        pipe.InitBuffer(inQueueGamma, 1, sizeof(float) * hLength);
        pipe.InitBuffer(outQueuePdX, 1, sizeof(float) * bshLength);
        pipe.InitBuffer(outQueueResForGamma, 1, sizeof(float) * bshLength);
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
        AscendC::LocalTensor<float> inputXLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> inputDyLocal = inQueueDy.AllocTensor<float>();
        AscendC::LocalTensor<float> inputVarianceLocal = inQueueVariance.AllocTensor<float>();
        AscendC::LocalTensor<float> inputMeanLocal = inQueueMean.AllocTensor<float>();
        AscendC::LocalTensor<float> inputGammaLocal = inQueueGamma.AllocTensor<float>();

        AscendC::DataCopy(inputXLocal, inputXGlobal, bshLength);
        AscendC::DataCopy(inputDyLocal, inputDyGlobal, bshLength);
        AscendC::DataCopy(inputVarianceLocal, inputVarianceGlobal, bsLength);
        AscendC::DataCopy(inputMeanLocal, inputMeanGlobal, bsLength);
        AscendC::DataCopy(inputGammaLocal, inputGammaGlobal, hLength);
        
        inQueueX.EnQue(inputXLocal);
        inQueueDy.EnQue(inputDyLocal);
        inQueueVariance.EnQue(inputVarianceLocal);
        inQueueMean.EnQue(inputMeanLocal);
        inQueueGamma.EnQue(inputGammaLocal);
    }
    
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> inputXLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> inputDyLocal = inQueueDy.DeQue<float>();
        AscendC::LocalTensor<float> inputVarianceLocal = inQueueVariance.DeQue<float>();
        AscendC::LocalTensor<float> inputMeanLocal = inQueueMean.DeQue<float>();
        AscendC::LocalTensor<float> inputGammaLocal = inQueueGamma.DeQue<float>();
        
        AscendC::LocalTensor<float> outputPdXLocal = outQueuePdX.AllocTensor<float>();
        AscendC::LocalTensor<float> outputResForGammaLocal = outQueueResForGamma.AllocTensor<float>();
        
        AscendC::LayerNormGrad<float, isReuseSource>(outputPdXLocal, outputResForGammaLocal,
                                                    inputDyLocal, inputXLocal, inputVarianceLocal, 
                                                    inputMeanLocal, inputGammaLocal, (float)epsilon, 
                                                    tiling_, {DataFormat::ND});
        
        outQueuePdX.EnQue(outputPdXLocal);
        outQueueResForGamma.EnQue(outputResForGammaLocal);
        
        inQueueX.FreeTensor(inputXLocal);
        inQueueDy.FreeTensor(inputDyLocal);
        inQueueVariance.FreeTensor(inputVarianceLocal);
        inQueueMean.FreeTensor(inputMeanLocal);
        inQueueGamma.FreeTensor(inputGammaLocal);
    }
    
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> outputPdXLocal = outQueuePdX.DeQue<float>();
        AscendC::LocalTensor<float> outputResForGammaLocal = outQueueResForGamma.DeQue<float>();
        
        AscendC::DataCopy(outputPdXGlobal, outputPdXLocal, bshLength);
        AscendC::DataCopy(outputResForGammaGlobal, outputResForGammaLocal, bshLength);
        
        outQueuePdX.FreeTensor(outputPdXLocal);
        outQueueResForGamma.FreeTensor(outputResForGammaLocal);
    }
    
private:
    AscendC::GlobalTensor<float> inputXGlobal;
    AscendC::GlobalTensor<float> inputDyGlobal;
    AscendC::GlobalTensor<float> inputVarianceGlobal;
    AscendC::GlobalTensor<float> inputMeanGlobal;
    AscendC::GlobalTensor<float> inputGammaGlobal;
    AscendC::GlobalTensor<float> outputPdXGlobal;
    AscendC::GlobalTensor<float> outputResForGammaGlobal;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueDy;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueVariance;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueMean;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueGamma;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueuePdX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueResForGamma;
    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;
    float epsilon;
    LayerNormGradTiling tiling_;
    uint32_t bshLength;
    uint32_t bsLength;
};
}

extern "C" __global__ __aicore__ void layernorm_grad_custom(GM_ADDR inputXGm, GM_ADDR inputDyGm,
                                                           GM_ADDR inputVarianceGm, GM_ADDR inputMeanGm, 
                                                           GM_ADDR inputGammaGm, GM_ADDR outputPdXGm, 
                                                           GM_ADDR resForGammaGm, GM_ADDR workspace, 
                                                           GM_ADDR tiling)
{
    if ASCEND_IS_AIC {
        return;
    }
    MyCustomKernel::VecTiling tilingData;
    CopyTiling(&tilingData, tiling);
    MyCustomKernel::KernelLayernormGrad<false> op;
    op.Init(inputXGm, inputDyGm, inputVarianceGm, inputMeanGm, inputGammaGm, outputPdXGm,
           resForGammaGm, tilingData);
    op.Process();
}
```

### 功能说明

LayerNormGrad Tiling的功能如下：

- **在host侧获取预留/申请的最大最小临时空间大小**：
  - kernel侧LayerNormGrad接口的计算需要开发者预留/申请临时空间
  - GetLayerNormGradMaxMinTmpSize接口用于在host侧获取预留/申请的最大最小临时空间大小
  - 开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用
  - 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小
  - 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请

- **通过GetLayerNormGradNDTilingInfo获取LayerNormGrad kernel侧接口所需tiling参数**：
  - 需要传入输入shape，剩余的可供LayerNormGrad接口计算的空间大小和计算的数据类型

LayerNormGrad Tiling结构体的定义如下，开发者无需关注该Tiling结构的具体信息，只需要传递到kernel侧，传入LayerNormGrad高阶API接口，直接进行使用即可：

```cpp
struct LayerNormGradTiling {
    uint32_t stackBufferSize = 0;
    uint32_t bLength = 0;
    uint32_t sLength = 0;
    uint32_t hLength = 0;
    uint32_t originalHLength = 0;
    uint32_t oneCalSize = 0;
    uint32_t nohCalSize = 0;
    uint32_t loopNum = 0;
    uint32_t tailSize = 0;
    uint32_t nohTailSize = 0;
    uint32_t tmpTensorBSHPos = 0;
    uint32_t tmpTensorBSHSize = 0;
    uint32_t pdVarTensorPos = 0;
    uint32_t pdVarTensorSize = 0;
    uint32_t pdMeanTensorPos = 0;
    uint32_t pdMeanTensorSize = 0;
    uint32_t x1TensorPos = 0;
    uint32_t x1TensorSize = 0;
    uint32_t x2TensorPos = 0;
    uint32_t x2TensorSize = 0;
    uint32_t x3TensorPos = 0;
    uint32_t x3TensorSize = 0;
    uint32_t tmpTensorPos = 0;
    uint32_t tmpTensorSize = 0;
    uint32_t tmpTensor1Pos = 0;
    uint32_t tmpTensor1Size = 0;
    uint32_t tmpTensor2Pos = 0;
    uint32_t tmpTensor2Size = 0;
    uint32_t lastDimValueBack = 0;
    uint32_t lastDimValueBackMulTwo = 0;
};
```

### 函数原型

```cpp
void GetLayerNormGradMaxMinTmpSize(const ge::Shape &srcShape, const uint32_t typeSize, 
                                  const bool isReuseSource, uint32_t &maxValue, uint32_t &minValue)

void GetLayerNormGradNDTilingInfo(const ge::Shape srcShape, const uint32_t stackBufferSize, 
                                 const uint32_t typeSize, const bool isReuseSource, 
                                 optiling::LayerNormGradTiling &tiling)
```

### 参数说明

#### 表 GetLayerNormGradMaxMinTmpSize 接口参数列表

| 参数名称 | 输入/输出 | 含义 |
|---------|-----------|------|
| srcShape | 输入 | 输入数据inputDy的shape信息{B, S, storageHLength, originHLength}，包括当前输入的inputDy的shape信息，以及地址对齐前（如存在H轴补齐操作）的原有shape信息。在API支持的场景下，storageHLength和originHLength保持一致 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2 |
| isReuseSource | 输入 | 是否复用源操作数的内存空间，与LayerNorm接口一致 |
| maxValue | 输出 | LayerNormGrad接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。<br>**说明**：maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小 |
| minValue | 输出 | LayerNormGrad接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间 |

#### 表 GetLayerNormGradNDTilingInfo 接口参数列表

| 参数名称 | 输入/输出 | 含义 |
|---------|-----------|------|
| srcShape | 输入 | 输入数据inputDy的shape信息，包括当前输入的shape信息，以及地址对齐前的原有shape信息 |
| stackBufferSize | 输入 | 可供接口使用的空间大小，单位元素个数 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2 |
| isReuseSource | 输入 | 是否可以复用inputX和inputDy的内存空间 |
| tiling | 输出 | 输入数据的切分信息 |

### 返回值说明
无

### 约束说明
无

### 调用示例

如下样例介绍了使用LayerNormGrad高阶API时host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中输入Tensor的shape大小为[2, 16, 64]，输入的数据类型为half。

#### 步骤1：将LayerNormGradTiling结构体参数增加至TilingData结构体

```cpp
BEGIN_TILING_DATA_DEF(TilingData) // 注册一个tiling的类，以tiling的名字作为入参
TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
TILING_DATA_FIELD_DEF(uint32_t, tileNum); // 添加tiling字段，每个核上总计算数据分块个数
... // 添加其他tiling字段
TILING_DATA_FIELD_DEF_STRUCT(LayerNormGradTiling, layernormGradTilingData); // 将LayerNormGradTiling结构体参数增加至TilingData结构体
END_TILING_DATA_DEF;
```

#### 步骤2：Tiling实现函数

```cpp
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    // 设置其他Tiling参数
    ...
    
    // {B, S, storageHLength, originHLength}
    std::vector<int64_t> shapeVec = {2, 16, 64, 64};
    ge::Shape srcShape(shapeVec);
    
    // 本样例中仅作为样例说明，通过GetLayerNormGradMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    uint32_t max;
    uint32_t min;
    AscendC::GetLayerNormGradMaxMinTmpSize(srcShape, sizeof(half), false, max, min);
    
    // 获取LayernormGrad Tiling参数
    AscendC::GetLayerNormGradNDTilingInfo(srcShape, min, sizeof(half), false, tiling.layernormGradTilingData);
    
    ... // 其他逻辑
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

#### 步骤3：对应的kernel侧实现

```cpp
extern "C" __global__ __aicore__ void func_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelFunc op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum, tilingData.layernormGradTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```
