project_json_src='''
[
    {
        "op": "MaskedCumsumCustom",
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
            },
            {
                "name": "mask",
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
BEGIN_TILING_DATA_DEF(MaskedCumsumCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rowCount);
TILING_DATA_FIELD_DEF(uint32_t, colCount);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaskedCumsumCustom, MaskedCumsumCustomTilingData)
}
"""

host_operator_src="""
#include "masked_cumsum_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    MaskedCumsumCustomTilingData tiling;
    const auto inputShape = context->GetInputTensor(0)->GetStorageShape();
    const size_t dimNum = inputShape.GetDimNum();
    if (dimNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t rowCount = 1;
    for (size_t i = 0; i + 1 < dimNum; ++i) {
        rowCount *= static_cast<uint32_t>(inputShape.GetDim(i));
    }
    const uint32_t colCount = static_cast<uint32_t>(inputShape.GetDim(dimNum - 1));
    if (colCount == 0) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_rowCount(rowCount);
    tiling.set_colCount(colCount);

    context->SetBlockDim(BLOCK_DIM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class MaskedCumsumCustom : public OpDef {
public:
    explicit MaskedCumsumCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mask")
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
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MaskedCumsumCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

class KernelMaskedCumsum {
public:
    __aicore__ inline KernelMaskedCumsum() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR y, uint32_t rowCount, uint32_t colCount)
    {
        this->rowCount = rowCount;
        this->colCount = colCount;

        xGm.SetGlobalBuffer((__gm__ float *)x, rowCount * colCount);
        maskGm.SetGlobalBuffer((__gm__ float *)mask, rowCount * colCount);
        yGm.SetGlobalBuffer((__gm__ float *)y, rowCount * colCount);
        pipe.InitBuffer(inQueueX, 1, colCount * sizeof(float));
        pipe.InitBuffer(inQueueMask, 1, colCount * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, colCount * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t rowIdx = 0; rowIdx < rowCount; ++rowIdx) {
            CopyIn(rowIdx);
            Compute();
            CopyOut(rowIdx);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> maskLocal = inQueueMask.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[rowIdx * colCount], colCount);
        AscendC::DataCopy(maskLocal, maskGm[rowIdx * colCount], colCount);
        inQueueX.EnQue(xLocal);
        inQueueMask.EnQue(maskLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> maskLocal = inQueueMask.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        float running = 0.0f;
        for (uint32_t i = 0; i < colCount; ++i) {
            const float maskValue = maskLocal.GetValue(i);
            if (maskValue != 0.0f) {
                running += xLocal.GetValue(i);
            }
            yLocal.SetValue(i, running);
        }
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueMask.FreeTensor(maskLocal);
    }

    __aicore__ inline void CopyOut(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[rowIdx * colCount], yLocal, colCount);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueMask;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> maskGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t rowCount;
    uint32_t colCount;
};

extern "C" __global__ __aicore__ void masked_cumsum_custom(
    GM_ADDR x,
    GM_ADDR mask,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaskedCumsum op;
    op.Init(x, mask, y, tiling_data.rowCount, tiling_data.colCount);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor masked_cumsum_impl_npu(const at::Tensor &x, const at::Tensor &mask)
{
    at::Tensor xContiguous = x.contiguous();
    at::Tensor maskFloat = mask.to(x.scalar_type()).contiguous();
    at::Tensor result = at::empty_like(xContiguous);
    EXEC_NPU_CMD(aclnnMaskedCumsumCustom, xContiguous, maskFloat, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("masked_cumsum_custom", &masked_cumsum_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("masked_cumsum_custom", &masked_cumsum_impl_npu, "masked cumsum on the last dimension");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.dim < 0:
            dim = x.dim() + self.dim
        else:
            dim = self.dim
        if dim != x.dim() - 1:
            raise RuntimeError("masked_cumsum_custom only supports the last dimension")
        return custom_ops_lib.masked_cumsum_custom(x, mask)
'''
