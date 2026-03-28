##### LayerNormGradBeta Tiling

```markdown
类型为 `LocalTensor`，支持的 `TPosition` 为 `VECIN`/`VECCALC`/`VECOUT`。

## 参数说明

| 参数名称 | 输入/输出 | 含义 |
|---------|-----------|------|
| tiling | 输入 | LayerNormGradBeta 计算所需 Tiling 信息，Tiling 信息的获取请参考 15.1.5.4.6 LayerNormGradBeta Tiling。 |

## 返回值说明
无

## 约束说明
- 操作数地址对齐要求请参见通用地址对齐约束。
- 源操作数和目的操作数的 Tensor 空间可以复用。
- 仅支持输入 shape 为 ND 格式。
- 输入数据不满足对齐要求时，开发者需要进行补齐，补齐的数据应设置为 0，防止出现异常值从而影响网络计算。
- 不支持对尾轴 H 轴的切分。

## 调用示例

```cpp
#include "kernel_operator.h"

template <typename T, bool isReuseSource = false>
class KernelLayernormGradBeta {
public:
    __aicore__ inline KernelLayernormGradBeta() {}
    
    __aicore__ inline void Init(__gm__ uint8_t *resForGammaGm, __gm__ uint8_t *inputDyGm,
                               __gm__ uint8_t *outputPdGammaGm, __gm__ uint8_t *outputPdBetaGm, 
                               const LayerNormGradBetaTiling &tiling) {
        this->bLength = tiling.bLength;
        this->sLength = tiling.sLength;
        this->hLength = tiling.hLength;
        this->tiling = tiling;
        bshLength = bLength * sLength * hLength;
        bsLength = bLength * sLength;
        
        resForGammaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(resForGammaGm), bshLength);
        inputDyGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputDyGm), bshLength);
        outputPdGammaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(outputPdGammaGm), hLength);
        outputPdBetaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(outputPdBetaGm), hLength);
        
        pipe.InitBuffer(inQueueResForGamma, 1, sizeof(T) * bshLength);
        pipe.InitBuffer(inQueueDy, 1, sizeof(T) * bshLength);
        pipe.InitBuffer(outQueuePdGamma, 1, sizeof(T) * hLength);
        pipe.InitBuffer(outQueuePdBeta, 1, sizeof(T) * hLength);
    }
    
    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        AscendC::LocalTensor<T> resForGammaLocal = inQueueResForGamma.AllocTensor<T>();
        AscendC::LocalTensor<T> inputDyLocal = inQueueDy.AllocTensor<T>();

        AscendC::DataCopy(resForGammaLocal, resForGammaGlobal, bshLength);
        AscendC::DataCopy(inputDyLocal, inputDyGlobal, bshLength);
        
        inQueueResForGamma.EnQue(resForGammaLocal);
        inQueueDy.EnQue(inputDyLocal);
    }
    
    __aicore__ inline void Compute() {
        AscendC::LocalTensor<T> resForGammaLocal = inQueueResForGamma.DeQue<T>();
        AscendC::LocalTensor<T> inputDyLocal = inQueueDy.DeQue<T>();
        AscendC::LocalTensor<T> outputPdGammaLocal = outQueuePdGamma.AllocTensor<T>();
        AscendC::LocalTensor<T> outputPdBetaLocal = outQueuePdBeta.AllocTensor<T>();

        AscendC::LayerNormGradBeta<T, isReuseSource>(
            outputPdGammaLocal, outputPdBetaLocal, resForGammaLocal, inputDyLocal, tiling);

        outQueuePdGamma.EnQue<T>(outputPdGammaLocal);
        outQueuePdBeta.EnQue<T>(outputPdBetaLocal);
        inQueueResForGamma.FreeTensor(resForGammaLocal);
        inQueueDy.FreeTensor(inputDyLocal);
    }
    
    __aicore__ inline void CopyOut() {
        AscendC::LocalTensor<T> outputPdGammaLocal = outQueuePdGamma.DeQue<T>();
        AscendC::LocalTensor<T> outputPdBetaLocal = outQueuePdBeta.DeQue<T>();
        
        AscendC::DataCopy(outputPdGammaGlobal, outputPdGammaLocal, hLength);
        AscendC::DataCopy(outputPdBetaGlobal, outputPdBetaLocal, hLength);
        
        outQueuePdGamma.FreeTensor(outputPdGammaLocal);
        outQueuePdBeta.FreeTensor(outputPdBetaLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueResForGamma, inQueueDy;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueuePdGamma, outQueuePdBeta;
    AscendC::GlobalTensor<T> resForGammaGlobal;
    AscendC::GlobalTensor<T> inputDyGlobal;
    AscendC::GlobalTensor<T> outputPdGammaGlobal;
    AscendC::GlobalTensor<T> outputPdBetaGlobal;
    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;
    uint32_t bshLength;
    uint32_t bsLength;
    LayerNormGradBetaTiling tiling;
};

extern "C" __global__ __aicore__ void kernel_layernorm_grad_beta_operator(
    GM_ADDR outputPdGammaGm, GM_ADDR outputPdBetaGm, GM_ADDR resForGammaGm, 
    GM_ADDR inputDyGm, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelLayernormGradBeta<half, false> op;
    op.Init(resForGammaGm, inputDyGm, outputPdGammaGm, outputPdBetaGm,
            tilingData.layerNormGradBetaTiling);
    op.Process();
}
```

## 15.1.5.4.6 LayerNormGradBeta Tiling

### 功能说明
LayerNormGradBeta Tiling 的功能如下：

- **在 host 侧获取预留/申请的最大最小临时空间大小**：
  kernel 侧 LayerNormGradBeta 接口的计算需要开发者预留/申请临时空间，`GetLayerNormGradBetaMaxMinTmpSize` 接口用于在 host 侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为 Tiling 参数传递到 kernel 侧使用。
  - 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
  - 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel 侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

- **通过 `GetLayerNormGradBetaNDTilingInfo` 获取 LayerNormGradBeta kernel 侧接口所需 tiling 参数**，需要传入输入 shape，剩余的可供 LayerNormGradBeta 接口计算的空间大小和计算的数据类型。

LayerNormGradBeta Tiling 结构体的定义如下，开发者无需关注该 Tiling 结构的具体信息，只需要传递到 kernel 侧，传入 LayerNormGradBeta 高阶 API 接口，直接进行使用即可。

```cpp
struct LayerNormGradBetaTiling {
    uint32_t stackBufferSize = 0;
    uint32_t bLength = 0;
    uint32_t sLength = 0;
    uint32_t hLength = 0;
    uint32_t originalHLength = 0;
    uint32_t bshLength = 0;
    uint32_t bsLength = 0;
    uint32_t oneCalSize = 0;
    uint32_t numberOfTmpBuf = 0;
    uint32_t loopRound = 0;
    uint32_t inputTailSize = 0;
    uint32_t inputTailPos = 0;
    uint32_t bsTailSize = 0;
    uint32_t bshCurLength = 0;
    uint32_t bsCurLength = 0;
    uint32_t gammaTempTensorPos = 0;
    uint32_t betaTempTensorPos = 0;
    uint32_t inputDyTmpTensorPos = 0;
    uint32_t resForGammaTmpTensorPos = 0;
    uint32_t reserved = 0;
};
```

### 函数原型
```cpp
void GetLayerNormGradBetaMaxMinTmpSize(const ge::Shape &srcShape, const uint32_t typeSize, 
                                      const bool isReuseSource, uint32_t &maxValue, uint32_t &minValue)
                                      
void GetLayerNormGradBetaNDTilingInfo(const ge::Shape srcShape, const uint32_t stackBufferSize, 
                                     const uint32_t typeSize, const bool isReuseSource, 
                                     optiling::LayerNormGradBetaTiling &tiling)
```

### 参数说明

#### 表 15-753 GetLayerNormGradBetaMaxMinTmpSize 接口参数列表

| 参数名称 | 输入/输出 | 含义 |
|---------|-----------|------|
| srcShape | 输入 | 输入数据 inputDy 的 shape 信息 {B, S, storageHLength, originHLength}，包括当前输入的 inputDy 的 shape 信息，以及地址对齐前（如存在 H 轴补齐操作）的原有 shape 信息。在 API 支持的场景下，storageHLength 和 originHLength 保持一致。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为 half，此处应传入 2。 |
| isReuseSource | 输入 | 是否复用源操作数的内存空间，与 LayerNorm 接口一致。 |
| maxValue | 输出 | LayerNormGradBeta 接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel 侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为 0 表示计算不需要临时空间。<br>**说明**：maxValue 仅作为参考值，有可能大于 Unified Buffer 剩余空间的大小，该场景下，开发者需要根据 Unified Buffer 剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | LayerNormGradBeta 接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为 0 表示计算不需要临时空间。 |

#### 表 15-754 GetLayerNormGradBetaNDTilingInfo 接口参数列表

| 参数名称 | 输入/输出 | 含义 |
|---------|-----------|------|
| srcShape | 输入 | 输入数据 inputDy 的 shape 信息，包括当前输入的 shape 信息，以及地址对齐前的原有 shape 信息。 |
| stackBufferSize | 输入 | 可供接口使用的空间大小，单位为元素个数。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为 half，此处应传入 2。 |
| isReuseSource | 输入 | 是否可以复用 inputDy 的内存空间。 |
| tiling | 输出 | 输入数据的切分信息。 |

### 返回值说明
无

### 约束说明
无

### 调用示例
如下样例介绍了使用 LayerNormGradBeta 高阶 API 时 host 侧获取 Tiling 参数的流程以及该参数如何在 kernel 侧使用。样例中输入 Tensor 的 shape 大小为 [2, 16, 64]，输入的数据类型为 half。

#### 步骤 1
将 LayerNormGradBetaTiling 结构体参数增加至 TilingData 结构体，作为 TilingData 结构体的一个字段。

```cpp
BEGIN_TILING_DATA_DEF(TilingData) // 注册一个 tiling 的类，以 tiling 的名字作为入参
    TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加 tiling 字段，总计算数据量
    TILING_DATA_FIELD_DEF(uint32_t, tileNum); // 添加 tiling 字段，每个核上总计算数据分块个数
    ... // 添加其他 tiling 字段
    TILING_DATA_FIELD_DEF_STRUCT(LayerNormGradBetaTiling, layernormGradBetaTilingData); // 将 LayerNormGradBetaTiling 结构体参数增加至 TilingData 结构体
END_TILING_DATA_DEF;
```

#### 步骤 2
Tiling 实现函数中，首先调用 `GetLayerNormGradBetaMaxMinTmpSize` 接口获取 LayerNormGradBeta 接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后调用 `GetLayerNormGradBetaNDTilingInfo` 接口根据输入 shape、剩余的可供计算的空间大小等信息获取 LayerNormGradBeta kernel 侧接口所需 tiling 参数。

```cpp
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    TilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    // 设置其他 Tiling 参数
    ...
    
    // {B, S, storageHLength, originHLength}
    std::vector<int64_t> shapeVec = {2, 16, 64, 64};
    ge::Shape srcShape(shapeVec);
    
    // 本样例中仅作为样例说明，通过 GetLayerNormGradBetaMaxMinTmpSize 获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    uint32_t max;
    uint32_t min;
    AscendC::GetLayerNormGradBetaMaxMinTmpSize(srcShape, sizeof(half), false, max, min);
    
    // 获取 LayerNormGradBeta Tiling 参数
    AscendC::GetLayerNormGradBetaNDTilingInfo(srcShape, min, sizeof(half), false, 
                                             tiling.layernormGradBetaTilingData);
    
    ... // 其他逻辑
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), 
                       context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

#### 步骤 3
对应的 kernel 侧通过在核函数中调用 `GET_TILING_DATA` 获取 TilingData，继而将 TilingData 中的 LayerNormGradBetaTiling 信息传入 LayerNormGradBeta 接口参与计算。完整的 kernel 侧样例请参考 15.1.5.4.5 LayerNormGradBeta。

```cpp
extern "C" __global__ __aicore__ void func_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
                                                 GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelFunc op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum, 
            tilingData.layernormGradBetaTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```
