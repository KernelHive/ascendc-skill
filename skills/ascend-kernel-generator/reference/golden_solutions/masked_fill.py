project_json_src = '''
[
    {
        "op": "MaskedFillCustom",
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

host_tiling_src = """
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MaskedFillCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaskedFillCustom, MaskedFillCustomTilingData)
}
"""

host_operator_src = """
#include "masked_fill_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 1;
const uint32_t TILE_NUM = 256;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MaskedFillCustomTilingData tiling;
    const uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class MaskedFillCustom : public OpDef {
public:
    explicit MaskedFillCustom(const char* name) : OpDef(name)
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

OP_ADD(MaskedFillCustom);
}
"""

kernel_src = """
#include "kernel_operator.h"
#include <math.h>

constexpr int32_t BUFFER_NUM = 1;

class KernelMaskedFill {
public:
    __aicore__ inline KernelMaskedFill() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR mask,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        this->totalLength = totalLength;
        this->tileLength = tileNum;
        this->tileCount = (totalLength + this->tileLength - 1) / this->tileLength;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalLength);
        maskGm.SetGlobalBuffer((__gm__ DTYPE_MASK*)mask, totalLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueMask, BUFFER_NUM, this->tileLength * sizeof(DTYPE_MASK));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileCount; ++i) {
            const uint32_t offset = i * this->tileLength;
            const uint32_t curLength = offset + this->tileLength <= this->totalLength
                ? this->tileLength
                : (this->totalLength - offset);
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_MASK> maskLocal = inQueueMask.AllocTensor<DTYPE_MASK>();
        AscendC::DataCopy(xLocal, xGm[offset], curLength);
        AscendC::DataCopy(maskLocal, maskGm[offset], curLength);
        inQueueX.EnQue(xLocal);
        inQueueMask.EnQue(maskLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_MASK> maskLocal = inQueueMask.DeQue<DTYPE_MASK>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        for (uint32_t i = 0; i < curLength; ++i) {
            const DTYPE_MASK maskValue = maskLocal.GetValue(i);
            yLocal.SetValue(i, maskValue != static_cast<DTYPE_MASK>(0) ? -INFINITY : xLocal.GetValue(i));
        }

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueMask.FreeTensor(maskLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[offset], yLocal, curLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueMask;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_MASK> maskGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t tileCount;
};

extern "C" __global__ __aicore__ void masked_fill_custom(
    GM_ADDR x,
    GM_ADDR mask,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaskedFill op;
    op.Init(x, mask, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
"""

python_bind_src = """
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor masked_fill_custom_impl_npu(const at::Tensor& x, const at::Tensor& mask)
{
    at::Tensor xContiguous = x.contiguous();
    at::Tensor maskFloat = mask.contiguous().to(at::kFloat);
    at::Tensor result = at::empty_like(xContiguous);
    EXEC_NPU_CMD(aclnnMaskedFillCustom, xContiguous, maskFloat, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("masked_fill_custom", &masked_fill_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("masked_fill_custom", &masked_fill_custom_impl_npu, "masked_fill(x, mask, -inf)");
}
"""

model_src = '''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.masked_fill_custom(x, mask)
'''
