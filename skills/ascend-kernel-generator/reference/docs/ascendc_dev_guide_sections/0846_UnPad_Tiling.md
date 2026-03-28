##### UnPad Tiling

## 返回值说明
无

## 约束说明
- 操作数地址对齐要求请参见通用地址对齐约束。

## 调用示例
本样例：Tensor的width已32B对齐，以half为例，如16*16，进行UnPad，变成16*15。输入数据类型均为half。

```cpp
#include "kernel_operator.h"

template <typename T>
class KernelUnPad {
public:
    __aicore__ inline KernelUnPad() {}
    
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR srcGm, uint16_t heightIn, uint16_t widthIn,
                                uint16_t oriWidthIn, AscendC::UnPadParams &unPadParamsIn, 
                                const UnPadTiling &tilingData)
    {
        height = heightIn;
        width = widthIn;
        oriWidth = oriWidthIn;
        unPadParams = unPadParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T *)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T *)dstGm);
        pipe.InitBuffer(inQueueSrcVecIn, 1, height * width * sizeof(T));
        pipe.InitBuffer(inQueueSrcVecOut, 1, height * (width - unPadParams.leftPad - unPadParams.rightPad) * sizeof(T));
        tiling = tilingData;
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
        AscendC::LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, srcGlobal, height * width);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> dstLocal = inQueueSrcVecIn.DeQue<T>();
        AscendC::LocalTensor<T> srcOutLocal = inQueueSrcVecOut.AllocTensor<T>();
        AscendC::UnPad(srcOutLocal, dstLocal, unPadParams, tiling);
        inQueueSrcVecOut.EnQue(srcOutLocal);
        inQueueSrcVecIn.FreeTensor(dstLocal);
    }
    
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> srcOutLocalDe = inQueueSrcVecOut.DeQue<T>();
        AscendC::DataCopy(dstGlobal, srcOutLocalDe, height * (width - unPadParams.leftPad - unPadParams.rightPad));
        inQueueSrcVecOut.FreeTensor(srcOutLocalDe);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> inQueueSrcVecOut;
    AscendC::GlobalTensor<T> srcGlobal;
    AscendC::GlobalTensor<T> dstGlobal;
    uint16_t height;
    uint16_t width;
    uint16_t oriWidth;
    AscendC::UnPadParams unPadParams;
    UnPadTiling tiling;
};

extern "C" __global__ __aicore__ void kernel_unpad_half_16_16_16(GM_ADDR src_gm, GM_ADDR dst_gm, __gm__ uint8_t *tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelUnPad<half> op;
    AscendC::UnPadParams unPadParams{0, 1};
    op.Init(dst_gm, src_gm, 16, 16, 16, unPadParams, tilingData.unpadTilingData);
    op.Process();
}
```

## 功能说明
用于获取UnPad Tiling参数。

## 函数原型
```cpp
void GetUnPadMaxMinTmpSize(const platform_ascendc::PlatformAscendC &ascendcPlatform, 
                          const ge::Shape &srcShape, const uint32_t typeSize, 
                          uint32_t &maxValue, uint32_t &minValue)

void UnPadTilingFunc(const ge::Shape srcShape, const uint32_t stackBufferSize, 
                    const uint32_t typeSize, optiling::UnPadTiling& tiling)
```

## 参数说明

### 表 15-848 GetUnPadMaxMinTmpSize 接口参数说明

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| ascendcPlatform | 输入 | 传入硬件平台的信息，PlatformAscendC定义请参见构造及析构函数。 |
| srcShape | 输入 | 输入Tensor的shape信息，shape为二维。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| maxValue | 输出 | UnPad接口能完成计算所需最大临时空间大小。UnPad接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。<br>**说明**：maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | UnPad接口能完成计算所需最小临时空间大小。Pad接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。 |

### 表 15-849 UnPadTilingFunc 接口参数说明

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| srcShape | 输入 | 输入Tensor的shape信息，shape为二维。 |
| stackBufferSize | 输入 | 可供UnPad接口计算的空间大小。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| tiling | 输出 | 输出UnPad接口所需的tiling信息。 |

## 返回值说明
无

## 约束说明
无

## 调用示例
如下样例介绍了使用UnPad高阶API时host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中原始shape的大小为[320, 64]，需要unpad的目标shape大小为[320, 63]，输入的数据类型为half。

### 步骤1
将UnPadTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

```cpp
BEGIN_TILING_DATA_DEF(TilingData) // 注册一个tiling的类，以tiling的名字作为入参
TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
TILING_DATA_FIELD_DEF(uint32_t, tileNum); // 添加tiling字段，每个核上总计算数据分块个数
... // 添加其他tiling字段
TILING_DATA_FIELD_DEF_STRUCT(UnPadTiling, unpadTilingData); // 将UnPadTiling结构体参数增加至TilingData结构体
END_TILING_DATA_DEF;
```

### 步骤2
Tiling实现函数中，首先调用GetUnPadMaxMinTmpSize接口获取UnPad接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小；然后根据输入shape、剩余的可供计算的空间大小等信息获取UnPad kernel侧接口所需tiling参数。

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
    
    std::vector<int64_t> shapeVec = {320,64};
    ge::Shape srcShape(shapeVec);
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    AscendC::GetUnPadMaxMinTmpSize(ascendcPlatform, srcShape, sizeof(half), maxValue, minValue);
    
    // 本样例中仅作为样例说明，获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    const uint32_t localWorkSpaceSize = minValue;
    AscendC::UnPadTilingFunc(srcShape, localWorkSpaceSize, sizeof(half), tiling.unpadTilingData);
    ...
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

### 步骤3
对应的kernel侧通过在核函数中调用GET_TILING_DATA获取TilingData，继而将TilingData中的UnPad Tiling信息传入UnPad接口参与计算。完整的kernel侧样例请参考调用示例。

```cpp
extern "C" __global__ __aicore__ void func_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelFunc op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum, tilingData.unpadTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```
