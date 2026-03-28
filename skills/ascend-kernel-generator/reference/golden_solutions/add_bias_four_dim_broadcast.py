project_json_src = '''
[
    {
        "op": "AddBiasBroadcastCustom",
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
                "name": "bias",
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
BEGIN_TILING_DATA_DEF(AddBiasBroadcastCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, channel);
    TILING_DATA_FIELD_DEF(uint32_t, height);
    TILING_DATA_FIELD_DEF(uint32_t, width);
    TILING_DATA_FIELD_DEF(uint32_t, hw);
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, pairsPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, remainderPairs);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddBiasBroadcastCustom, AddBiasBroadcastCustomTilingData)
}
"""

host_operator_src = """
#include <algorithm>
#include <cstdint>
#include "add_bias_broadcast_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* biasShape = context->GetInputShape(1);
    const auto& xStorage = xShape->GetStorageShape();
    const auto& biasStorage = biasShape->GetStorageShape();

    if (xStorage.GetDimNum() != 4 || biasStorage.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batch = static_cast<uint32_t>(xStorage.GetDim(0));
    const uint32_t channel = static_cast<uint32_t>(xStorage.GetDim(1));
    const uint32_t height = static_cast<uint32_t>(xStorage.GetDim(2));
    const uint32_t width = static_cast<uint32_t>(xStorage.GetDim(3));
    const uint32_t biasChannel = static_cast<uint32_t>(biasStorage.GetDim(0));
    if (biasChannel != channel) {
        return ge::GRAPH_FAILED;
    }

    AddBiasBroadcastCustomTilingData tiling;
    const uint32_t hw = height * width;
    const uint32_t totalLength = batch * channel * hw;
    const uint32_t totalPairs = batch * channel;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim = ascendcPlatform.GetCoreNumAiv();
    if (blockDim == 0) {
        blockDim = 1;
    }
    if (totalPairs < blockDim) {
        blockDim = totalPairs;
    }
    if (blockDim == 0) {
        blockDim = 1;
    }

    tiling.set_batch(batch);
    tiling.set_channel(channel);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_hw(hw);
    tiling.set_totalLength(totalLength);
    tiling.set_pairsPerCore(totalPairs / blockDim);
    tiling.set_remainderPairs(totalPairs % blockDim);

    context->SetBlockDim(blockDim);
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
class AddBiasBroadcastCustom : public OpDef {
public:
    explicit AddBiasBroadcastCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
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

OP_ADD(AddBiasBroadcastCustom);
}
"""

kernel_src = """
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

class KernelAddBiasFourDimBroadcast {
public:
    __aicore__ inline KernelAddBiasFourDimBroadcast() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t channel,
        uint32_t hw,
        uint32_t totalLength,
        uint32_t pairsPerCore,
        uint32_t remainderPairs)
    {
        this->channel = channel;
        this->hw = hw;
        this->pairsPerCore = pairsPerCore;
        this->remainderPairs = remainderPairs;

        const uint32_t blockIdx = GetBlockIdx();
        this->pairCount = this->pairsPerCore + (blockIdx < this->remainderPairs ? 1U : 0U);
        this->startPair = blockIdx * this->pairsPerCore + (blockIdx < this->remainderPairs ? blockIdx : this->remainderPairs);

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalLength);
        biasGm.SetGlobalBuffer((__gm__ DTYPE_BIAS*)bias, channel);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->hw * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->hw * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->pairCount; ++i) {
            const uint32_t pairIdx = this->startPair + i;
            const uint32_t channelIdx = pairIdx % this->channel;
            const uint32_t offset = pairIdx * this->hw;
            CopyIn(offset);
            Compute(channelIdx);
            CopyOut(offset);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[offset], this->hw);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t channelIdx)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        const float biasValue = biasGm.GetValue(channelIdx);
        Adds(yLocal, xLocal, biasValue, this->hw);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset)
    {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[offset], yLocal, this->hw);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_BIAS> biasGm;
    GlobalTensor<DTYPE_Y> yGm;
    uint32_t channel = 0;
    uint32_t hw = 0;
    uint32_t pairsPerCore = 0;
    uint32_t remainderPairs = 0;
    uint32_t startPair = 0;
    uint32_t pairCount = 0;
};

extern "C" __global__ __aicore__ void add_bias_broadcast_custom(
    GM_ADDR x,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelAddBiasFourDimBroadcast op;
    op.Init(
        x,
        bias,
        y,
        tiling_data.channel,
        tiling_data.hw,
        tiling_data.totalLength,
        tiling_data.pairsPerCore,
        tiling_data.remainderPairs);
    op.Process();
}
"""

python_bind_src = """
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor add_bias_broadcast_custom_impl_npu(const at::Tensor& x, const at::Tensor& bias) {
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnAddBiasBroadcastCustom, x, bias, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("add_bias_broadcast_custom", &add_bias_broadcast_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_bias_broadcast_custom", &add_bias_broadcast_custom_impl_npu, "add bias over 4D tensor");
}
"""

model_src = '''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, bias):
        return custom_ops_lib.add_bias_broadcast_custom(x, bias)
'''
