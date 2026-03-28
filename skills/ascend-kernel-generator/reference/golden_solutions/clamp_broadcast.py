project_json_src='''
[
    {
        "op": "ClampBroadcastCustom",
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
                "name": "min_val",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "max_val",
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
BEGIN_TILING_DATA_DEF(ClampBroadcastCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ClampBroadcastCustom, ClampBroadcastCustomTilingData)
}
"""

host_operator_src="""
#include "clamp_broadcast_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 16;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    ClampBroadcastCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x_dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ClampBroadcastCustom : public OpDef {
public:
    explicit ClampBroadcastCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("min_val")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("max_val")
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

OP_ADD(ClampBroadcastCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelClampBroadcast {
public:
    __aicore__ inline KernelClampBroadcast() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR minVal,
        GM_ADDR maxVal,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        minValGm.SetGlobalBuffer((__gm__ float *)minVal + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        maxValGm.SetGlobalBuffer((__gm__ float *)maxVal + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueMin, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueMax, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> minLocal = inQueueMin.AllocTensor<float>();
        AscendC::LocalTensor<float> maxLocal = inQueueMax.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(minLocal, minValGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(maxLocal, maxValGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueMin.EnQue(minLocal);
        inQueueMax.EnQue(maxLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> minLocal = inQueueMin.DeQue<float>();
        AscendC::LocalTensor<float> maxLocal = inQueueMax.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();

        AscendC::Max(tmpLocal, xLocal, minLocal, this->tileLength);
        AscendC::Min(yLocal, tmpLocal, maxLocal, this->tileLength);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueMin.FreeTensor(minLocal);
        inQueueMax.FreeTensor(maxLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueMin;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueMax;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> minValGm;
    AscendC::GlobalTensor<float> maxValGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void clamp_broadcast_custom(
    GM_ADDR x,
    GM_ADDR min_val,
    GM_ADDR max_val,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelClampBroadcast op;
    op.Init(x, min_val, max_val, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor clamp_broadcast_impl_npu(
    const at::Tensor &self,
    const at::Tensor &min_val,
    const at::Tensor &max_val)
{
    at::Tensor min_tensor = min_val.expand(self.sizes()).contiguous();
    at::Tensor max_tensor = max_val.expand(self.sizes()).contiguous();
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnClampBroadcastCustom, self, min_tensor, max_tensor, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("clamp_broadcast_custom", &clamp_broadcast_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("clamp_broadcast_custom", &clamp_broadcast_impl_npu, "Broadcast clamp");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.clamp_broadcast_custom(x, min_val, max_val)
'''
