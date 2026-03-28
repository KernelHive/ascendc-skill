##### ASCENDC_TPL_SEL_PARAM

## 功能说明

Tiling模板编程时，开发者通过调用此接口自动生成并配置TilingKey。

使用该接口需要包含定义模板参数和模板参数组合的头文件。详细内容请参考Tiling模板编程。

## 函数原型

```c
#define ASCENDC_TPL_SEL_PARAM(context, ...) \
do { \
    uint64_t key = GET_TPL_TILING_KEY({__VA_ARGS__}); \
    context->SetTilingKey(key); \
} while(0)
```

> **说明**：context指代`TilingFunc(gert::TilingContext *context)`中的context

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| context | 输入 | TilingFunc注册上下文 |
| ... | 输入 | 可变长参数，模板参数的具体值，传入时需要与定义模板参数和模板参数组合的头文件中的模板参数顺序保持一致 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```c
#include "tiling_key_add_custom.h"

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
    ge::DataType dtype_y = context->GetInputDesc(1)->GetDataType();
    ge::DataType dtype_z = context->GetOutputDesc(1)->GetDataType();
    
    uint32_t D_T_X = ADD_TPL_FP32, D_T_Y = ADD_TPL_FP32, D_T_Z = ADD_TPL_FP32, 
             TILE_NUM = 1, IS_SPLIT = 0;
    
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
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), 
                       context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    ASCENDC_TPL_SEL_PARAM(context, D_T_X, D_T_Y, D_T_Z, TILE_NUM, IS_SPLIT);
    
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    
    return ge::GRAPH_SUCCESS;
}
```
