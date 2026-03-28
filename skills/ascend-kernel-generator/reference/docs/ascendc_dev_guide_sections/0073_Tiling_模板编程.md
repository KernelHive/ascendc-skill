#### Tiling 模板编程

在TilingKey编程方式中，TilingKey不易于记忆和理解，因为它们通常是较长又没有明确含义的数字。

在涉及多个TilingKey的场景中，开发者依赖TilingKey来管理kernel的实现，无论是在管理还是使用上都会遇到相当大的复杂性。为了简化这一过程，可以采用模板编程的方法来替代传统的TilingKey编程，从而减少对TilingKey数值标识的依赖，使kernel的管理更加直观和高效。

## 使用步骤

### 步骤1：定义模板参数和模板参数组合的头文件

在自定义算子工程的 `op_kernel` 目录下，新增定义模板参数和模板参数组合的头文件，本示例中头文件命名为 `tiling_key_add_custom.h`。

- 该头文件中需要包含模板头文件 `ascendc/host_api/tiling/template_argument.h`。
- 定义模板参数 `ASCENDC_TPL_ARGS_DECL` 和模板参数组合 `ASCENDC_TPL_ARGS_SEL`（即可使用的模板）。具体API参考见“模板参数定义”。

```cpp
#include "ascendc/host_api/tiling/template_argument.h"

#define ADD_TPL_FP16 1 // 数据类型定义
#define ADD_TPL_FP32 0

#define ADD_TPL_ND 2 // 数据格式定义
#define ADD_TPL_NZ 29

// 模板参数
ASCENDC_TPL_ARGS_DECL(AddTemplateCustom, // 算子OpType
ASCENDC_TPL_DTYPE_DECL(D_T_X, ADD_TPL_FP16, ADD_TPL_FP32), // DataType类型的模板参数定义：输入参数x的数据类型，取值范围为float16/float32
ASCENDC_TPL_DTYPE_DECL(D_T_Y, ADD_TPL_FP16, ADD_TPL_FP32), // DataType类型的模板参数定义：输入参数y的数据类型，取值范围为float16/float32
ASCENDC_TPL_DTYPE_DECL(D_T_Z, ADD_TPL_FP16, ADD_TPL_FP32), // DataType类型的模板参数定义：输入参数z的数据类型，取值范围为float16/float32
ASCENDC_TPL_UINT_DECL(TILE_NUM, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_MIX, 2, 0, 2, 3, 5, 10, 12, 13, 9, 8),// 自定义UINT类型（无符号整形）的模板参数定义：模板参数为切分的块数，编码位宽为ASCENDC_TPL_8_BW即8比特，表示该模板参数的个数不超过8比特能表达的范围；ASCENDC_TPL_UI_MIX表示通过混合模式表达取值范围，有2组的数据{0-2}、{3-5}和穷举值10、12、13、9、8，最后结果为{0, 1, 2, 3, 4, 5, 10, 12, 13, 9, 8}
ASCENDC_TPL_BOOL_DECL(IS_SPLIT, 0, 1), // 自定义bool类型的模板参数定义：模板参数为是否切分标志位，取值范围为0和1，1表示切分，0表示不切分
);

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
ASCENDC_TPL_ARGS_SEL(
ASCENDC_TPL_DTYPE_SEL(D_T_X, ADD_TPL_FP16),
ASCENDC_TPL_DTYPE_SEL(D_T_Y, ADD_TPL_FP16),
ASCENDC_TPL_DTYPE_SEL(D_T_Z, ADD_TPL_FP16),
ASCENDC_TPL_UINT_SEL(TILE_NUM, ASCENDC_TPL_UI_LIST, 1, 8),
ASCENDC_TPL_BOOL_SEL(IS_SPLIT, 0, 1)
),
ASCENDC_TPL_ARGS_SEL(
ASCENDC_TPL_DTYPE_SEL(D_T_X, ADD_TPL_FP32),
ASCENDC_TPL_DTYPE_SEL(D_T_Y, ADD_TPL_FP32),
ASCENDC_TPL_DTYPE_SEL(D_T_Z, ADD_TPL_FP32),
ASCENDC_TPL_UINT_SEL(TILE_NUM, ASCENDC_TPL_UI_LIST, 1, 8),
ASCENDC_TPL_BOOL_SEL(IS_SPLIT, 0, 1)
),
);
```

### 步骤2：host侧调用 GET_TPL_TILING_KEY 接口生成 TilingKey

- host实现文件中包含步骤1中定义模板参数和模板参数组合的头文件。
- 调用 `GET_TPL_TILING_KEY` 接口生成 TilingKey，`GET_TPL_TILING_KEY` 输入参数为模板参数的具体值，传入时需要与定义模板参数和模板参数组合的头文件中的模板参数顺序保持一致。

```cpp
#include "tiling_key_add_custom.h"
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
TilingData tiling;
uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
ge::DataType dtype_y = context->GetInputDesc(1)->GetDataType();
ge::DataType dtype_z = context->GetOutputDesc(1)->GetDataType();
uint32_t D_T_X = ADD_TPL_FP32, D_T_Y=ADD_TPL_FP32, D_T_Z=ADD_TPL_FP32, TILE_NUM=1, IS_SPLIT=0;

if(dtype_x == ge::DataType::DT_FLOAT){
D_T_X = ADD_TPL_FP32;
}else if(dtype_x == ge::DataType::DT_FLOAT16){
D_T_X = ADD_TPL_FP16;
}
if(dtype_y == ge::DataType::DT_FLOAT){
D_T_Y = ADD_TPL_FP32;
}else if(dtype_y == ge::DataType::DT_FLOAT16){
D_T_Y = ADD_TPL_FP16;
}
if(dtype_z == ge::DataType::DT_FLOAT){
D_T_Z = ADD_TPL_FP32;
}else if(dtype_z == ge::DataType::DT_FLOAT16){
D_T_Z = ADD_TPL_FP16;
}
if(totalLength< MIN_LENGTH_FOR_SPLIT){
IS_SPLIT = 0;
TILE_NUM = 1;
}else{
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

### 步骤3：kernel侧实现

- host实现文件中包含步骤1中定义模板参数和模板参数组合的头文件。
- 核函数添加 `template` 模板，以便支持模板参数的传入，参数顺序需要与定义模板参数和模板参数组合的头文件中的模板参数顺序保持一致。
- 通过对模板参数的分支判断，选择不同的kernel侧实现。

```cpp
#include "tiling_key_add_custom.h"
...
...
template<int D_T_X, int D_T_Y, int D_T_Z, int TILE_NUM, int IS_SPLIT>
__global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
GET_TILING_DATA(tiling_data, tiling);
if(D_T_X == ADD_TPL_FP32 && D_T_Y == ADD_TPL_FP32 && D_T_Z == ADD_TPL_FP32){
KernelAdd<float, float, float> op;
op.Init(x, y, z, tiling_data.totalLength, TILE_NUM, IS_SPLIT);
op.Process1();
}else if(D_T_X == ADD_TPL_FP16 && D_T_Y == ADD_TPL_FP16 && D_T_Z == ADD_TPL_FP16){
KernelAdd<half, half, half> op;
if(IS_SPLIT == 0){
op.Init(x, y, z, tiling_data.totalLength, TILE_NUM, IS_SPLIT);
op.Process1();
}else if(IS_SPLIT==1){
op.Init(x, y, z, tiling_data.totalLength, TILE_NUM, IS_SPLIT);
op.Process2();
}
}
}
```

## 说明

完整样例请参考 Tiling 模板编程样例。
