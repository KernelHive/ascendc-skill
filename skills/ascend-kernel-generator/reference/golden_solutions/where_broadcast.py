project_json_src='''
[
    {
        "op": "WhereBroadcastCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "cond",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
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
                "name": "y",
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
                "name": "z",
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
BEGIN_TILING_DATA_DEF(WhereBroadcastCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(WhereBroadcastCustom, WhereBroadcastCustomTilingData)
}
"""

host_operator_src="""
#include "where_broadcast_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    WhereBroadcastCustomTilingData tiling;
    const uint32_t totalLength = context->GetInputShape(1)->GetOriginShape().GetShapeSize();

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
    const gert::Shape* xShape = context->GetInputShape(1);
    gert::Shape* zShape = context->GetOutputShape(0);
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(1);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class WhereBroadcastCustom : public OpDef {
public:
    explicit WhereBroadcastCustom(const char* name) : OpDef(name)
    {
        this->Input("cond")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(WhereBroadcastCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelWhereBroadcast {
public:
    __aicore__ inline KernelWhereBroadcast() {}

    __aicore__ inline void Init(
        GM_ADDR cond,
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        this->totalLength = totalLength;
        this->tileLength = tileNum;
        this->tileCount = (totalLength + this->tileLength - 1) / this->tileLength;
        condGm.SetGlobalBuffer((__gm__ DTYPE_COND*)cond, totalLength);
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z*)z, totalLength);

        pipe.InitBuffer(inQueueCond, BUFFER_NUM, this->tileLength * sizeof(DTYPE_COND));
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
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
        AscendC::LocalTensor<DTYPE_COND> condLocal = inQueueCond.AllocTensor<DTYPE_COND>();
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(condLocal, condGm[offset], curLength);
        AscendC::DataCopy(xLocal, xGm[offset], curLength);
        AscendC::DataCopy(yLocal, yGm[offset], curLength);
        inQueueCond.EnQue(condLocal);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength)
    {
        AscendC::LocalTensor<DTYPE_COND> condLocal = inQueueCond.DeQue<DTYPE_COND>();
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();

        for (uint32_t i = 0; i < curLength; ++i) {
            const DTYPE_COND condValue = condLocal.GetValue(i);
            zLocal.SetValue(i, condValue != static_cast<DTYPE_COND>(0) ? xLocal.GetValue(i) : yLocal.GetValue(i));
        }

        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueCond.FreeTensor(condLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[offset], zLocal, curLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueCond;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_COND> condGm;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t tileCount;
};

extern "C" __global__ __aicore__ void where_broadcast_custom(
    GM_ADDR cond,
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelWhereBroadcast op;
    op.Init(cond, x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
"""

python_bind_src="""
#include <vector>
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor where_broadcast_custom_impl_npu(
    const at::Tensor& cond,
    const at::Tensor& x,
    const at::Tensor& y)
{
    std::vector<at::Tensor> xyBroadcasted = at::broadcast_tensors({x, y});
    at::Tensor xExpanded = xyBroadcasted[0].contiguous();
    at::Tensor yExpanded = xyBroadcasted[1].contiguous();
    at::Tensor condExpanded = cond.expand(xExpanded.sizes()).contiguous().to(at::kFloat);
    at::Tensor result = at::empty_like(xExpanded);
    EXEC_NPU_CMD(aclnnWhereBroadcastCustom, condExpanded, xExpanded, yExpanded, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("where_broadcast_custom", &where_broadcast_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("where_broadcast_custom", &where_broadcast_custom_impl_npu, "Broadcast where on expanded tensors");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, cond: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.where_broadcast_custom(cond, x, y)
'''
