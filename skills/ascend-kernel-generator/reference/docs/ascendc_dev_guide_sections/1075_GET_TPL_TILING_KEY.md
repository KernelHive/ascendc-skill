##### GET_TPL_TILING_KEY

## 功能说明

Tiling模板编程时，开发者通过调用此接口自动生成TilingKey。该接口将传入的模板参数通过定义的位宽，转成二进制，按照顺序组合后转成uint64数值，即TilingKey。

使用该接口需要包含定义模板参数和模板参数组合的头文件。详细内容请参考Tiling模板编程。

## 函数原型

```cpp
namespace AscendC {
uint64_t EncodeTilingKey(TilingDeclareParams declareParams,
                         TilingSelectParams selectParamsVec,
                         std::vector<uint64_t> tilingParams);
}

#define GET_TPL_TILING_KEY(...) \
    AscendC::EncodeTilingKey(g_tilingDeclareParams, g_tilingSelectParams, {__VA_ARGS__})
```

GET_TPL_TILING_KEY通过调用EncodeTilingKey接口生成TilingKey，EncodeTilingKey属于内部关联接口，开发者无需关注。

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ... | 输入 | 可变长参数，模板参数的具体值，传入时需要与定义模板参数和模板参数组合的头文件中的模板参数顺序保持一致 |

## 返回值说明

TilingKey数值。

## 约束说明

无。

## 调用示例

```cpp
#include "tiling_key_add_custom.h"

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
    ge::DataType dtype_y = context->GetInputDesc(1)->GetDataType();
    ge::DataType dtype_z = context->GetOutputDesc(1)->GetDataType();
    
    uint32_t D_T_X = ADD_TPL_FP32, D_T_Y = ADD_TPL_FP32, D_T_Z = ADD_TPL_FP32, TILE_NUM = 1, IS_SPLIT = 0;
    
    if (dtype_x == ge::DataType::DT_FLOAT) {
        D_T_X = ADD_TPL_FP32;
    } else if (dtype_x == ge::DataType::DT_FLOAT16) {
        D_T_X = ADD_TPL_FP16;
    }
    
    if (dtype_y == ge::DataType::DT_FLOAT) {
        D_T_Y = ADD_TPL_FP32;
    } else if (dtype_y == ge::DataType::DT_FLOAT16) {
        D_T_Y = ADD_TPL_FP16;
    }
    
    if (dtype_z == ge::DataType::DT_FLOAT) {
        D_T_Z = ADD_TPL_FP32;
    } else if (dtype_z == ge::DataType::DT_FLOAT16) {
        D_T_Z = ADD_TPL_FP16;
    }
    
    if (totalLength < MIN_LENGTH_FOR_SPLIT) {
        IS_SPLIT = 0;
        TILE_NUM = 1;
    } else {
        IS_SPLIT = 1;
        TILE_NUM = DEFAULT_TILE_NUM;
    }
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    const uint64_t tilingKey = GET_TPL_TILING_KEY(D_T_X, D_T_Y, D_T_Z, TILE_NUM, IS_SPLIT);
    context->SetTilingKey(tilingKey);
    
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    
    return ge::GRAPH_SUCCESS;
}
```
