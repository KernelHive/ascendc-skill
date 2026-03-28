project_json_src='''
[
    {
        "op": "LandBcCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "int16"
                ]
            },
            {
                "name": "y",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "int16"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "z",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "int16"
                ]
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LandBcCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LandBcCustom, LandBcCustomTilingData)
}
"""

host_operator_src="""
#include "land_bc_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 1;
constexpr uint32_t TILE_LENGTH = 1024;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LandBcCustomTilingData tiling;
    const uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(TILE_LENGTH);
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
    gert::Shape* zShape = context->GetOutputShape(0);
    *zShape = *xShape;
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
class LandBcCustom : public OpDef {
public:
    explicit LandBcCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LandBcCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelLandBc {
public:
    __aicore__ inline KernelLandBc() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength,
        uint32_t tileLength)
    {
        this->totalLength = totalLength;
        this->tileLength = tileLength;
        this->tileCount = (totalLength + tileLength - 1) / tileLength;

        xGm.SetGlobalBuffer((__gm__ int16_t*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ int16_t*)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ int16_t*)z, totalLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(int16_t));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(int16_t));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(int16_t));
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
        AscendC::LocalTensor<int16_t> xLocal = inQueueX.AllocTensor<int16_t>();
        AscendC::LocalTensor<int16_t> yLocal = inQueueY.AllocTensor<int16_t>();
        AscendC::DataCopy(xLocal, xGm[offset], curLength);
        AscendC::DataCopy(yLocal, yGm[offset], curLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength)
    {
        AscendC::LocalTensor<int16_t> xLocal = inQueueX.DeQue<int16_t>();
        AscendC::LocalTensor<int16_t> yLocal = inQueueY.DeQue<int16_t>();
        AscendC::LocalTensor<int16_t> zLocal = outQueueZ.AllocTensor<int16_t>();
        AscendC::And(zLocal, xLocal, yLocal, curLength);
        outQueueZ.EnQue<int16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength)
    {
        AscendC::LocalTensor<int16_t> zLocal = outQueueZ.DeQue<int16_t>();
        AscendC::DataCopy(zGm[offset], zLocal, curLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<int16_t> xGm;
    AscendC::GlobalTensor<int16_t> yGm;
    AscendC::GlobalTensor<int16_t> zGm;
    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t tileCount;
};

extern "C" __global__ __aicore__ void land_bc_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelLandBc op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileLength);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor land_bc_custom_impl_npu(const at::Tensor& x, const at::Tensor& y)
{
    at::Tensor xInt = x.to(at::kShort);
    at::Tensor yExpanded = y.expand(x.sizes()).contiguous().to(at::kShort);
    at::Tensor zInt = at::empty_like(xInt);
    EXEC_NPU_CMD(aclnnLandBcCustom, xInt, yExpanded, zInt);
    return zInt.ne(0);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("land_bc_custom", &land_bc_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("land_bc_custom", &land_bc_custom_impl_npu, "logical and with broadcast");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.land_bc_custom(a, b)
'''
