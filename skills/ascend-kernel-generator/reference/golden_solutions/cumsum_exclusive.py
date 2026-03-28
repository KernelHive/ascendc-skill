project_json_src='''
[
    {
        "op": "CumsumExclusiveCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "y",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CumsumExclusiveTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, outerDimsSize);
  TILING_DATA_FIELD_DEF(uint32_t, innerDimsSize);
  TILING_DATA_FIELD_DEF(uint32_t, axisSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CumsumExclusiveCustom, CumsumExclusiveTilingData)
}
"""

host_operator_src="""
#include "cumsum_exclusive_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CumsumExclusiveTilingData tiling;
    auto inputShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t totalLength = inputShape.GetShapeSize();
    uint32_t axis = 1;
    uint32_t outerDimsSize = 1;
    uint32_t innerDimsSize = 1;
    uint32_t axisSize = inputShape[axis];

    for (int i = 0; i < axis; i++) {
        outerDimsSize *= inputShape[i];
    }
    for (int i = axis + 1; i < inputShape.GetDimNum(); i++) {
        innerDimsSize *= inputShape[i];
    }

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_outerDimsSize(outerDimsSize);
    tiling.set_innerDimsSize(innerDimsSize);
    tiling.set_axisSize(axisSize);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class CumsumExclusiveCustom : public OpDef {
public:
    explicit CumsumExclusiveCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(CumsumExclusiveCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

class KernelCumsumExclusive {
public:
    __aicore__ inline KernelCumsumExclusive() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                uint32_t outerDimsSize, uint32_t innerDimsSize, uint32_t axisSize)
    {
        this->totalLength = totalLength;
        this->outerDimsSize = outerDimsSize;
        this->innerDimsSize = innerDimsSize;
        this->axisSize = axisSize;

        xGm.SetGlobalBuffer((__gm__ float *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalLength);

        pipe.InitBuffer(localInput, axisSize * sizeof(float));
        pipe.InitBuffer(localOutput, axisSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t outer = 0; outer < outerDimsSize; outer++) {
            for (uint32_t inner = 0; inner < innerDimsSize; inner++) {
                uint32_t baseOffset = outer * axisSize * innerDimsSize + inner;

                AscendC::LocalTensor<float> sliceX = localInput.Get<float>();
                AscendC::LocalTensor<float> sliceY = localOutput.Get<float>();

                for (uint32_t ax = 0; ax < axisSize; ax++) {
                    uint32_t idx = baseOffset + ax * innerDimsSize;
                    float val = xGm.GetValue(idx);
                    sliceX.SetValue(ax, val);
                }

                float cumsum = 0.0f;
                for (uint32_t ax = 0; ax < axisSize; ax++) {
                    float currentValue = sliceX.GetValue(ax);
                    sliceY.SetValue(ax, cumsum);
                    cumsum += currentValue;
                }

                for (uint32_t ax = 0; ax < axisSize; ax++) {
                    uint32_t idx = baseOffset + ax * innerDimsSize;
                    float result = sliceY.GetValue(ax);
                    yGm.SetValue(idx, result);
                }
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> localInput;
    AscendC::TBuf<AscendC::TPosition::VECCALC> localOutput;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalLength;
    uint32_t outerDimsSize;
    uint32_t innerDimsSize;
    uint32_t axisSize;
};

extern "C" __global__ __aicore__ void cumsum_exclusive_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelCumsumExclusive op;
    op.Init(x, y, tiling_data.totalLength,
            tiling_data.outerDimsSize, tiling_data.innerDimsSize, tiling_data.axisSize);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor cumsum_exclusive_custom_impl_npu(const at::Tensor& self) {
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnCumsumExclusiveCustom, self, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("cumsum_exclusive_custom", &cumsum_exclusive_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum_exclusive_custom", &cumsum_exclusive_custom_impl_npu, "exclusive cumulative sum");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib
class ModelNew(torch.nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return custom_ops_lib.cumsum_exclusive_custom(x)
'''
