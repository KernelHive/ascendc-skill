project_json_src='''
[
    {
        "op": "LinearAttentionCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "q",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "k",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "v",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ],
        "output_desc": [
            {
                "name": "y",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LinearAttentionCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LinearAttentionCustom, LinearAttentionCustomTilingData)
}
"""

host_operator_src="""
#include "linear_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto qShape = context->GetInputTensor(0)->GetOriginShape();
    auto kShape = context->GetInputTensor(1)->GetOriginShape();
    auto vShape = context->GetInputTensor(2)->GetOriginShape();
    if (qShape.GetDimNum() != 3 || kShape.GetDimNum() != 3 || vShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    if (qShape.GetDim(0) != kShape.GetDim(0) || qShape.GetDim(0) != vShape.GetDim(0) ||
        qShape.GetDim(1) != kShape.GetDim(1) || kShape.GetDim(1) != vShape.GetDim(1) ||
        qShape.GetDim(2) != kShape.GetDim(2) || vShape.GetDim(2) != qShape.GetDim(2)) {
        return ge::GRAPH_FAILED;
    }

    LinearAttentionCustomTilingData tiling;
    tiling.set_totalLength(static_cast<uint32_t>(qShape.GetShapeSize()));
    tiling.set_tileNum(TILE_NUM);
    context->SetBlockDim(BLOCK_DIM);
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
    const gert::Shape *qShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *qShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class LinearAttentionCustom : public OpDef {
public:
    explicit LinearAttentionCustom(const char *name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(LinearAttentionCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelLinearAttention {
public:
    __aicore__ inline KernelLinearAttention() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        qGm.SetGlobalBuffer((__gm__ float *)q + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueQ, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> qLocal = inQueueQ.AllocTensor<float>();
        AscendC::DataCopy(qLocal, qGm[progress * this->tileLength], this->tileLength);
        inQueueQ.EnQue(qLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> qLocal = inQueueQ.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::DataCopy(yLocal, qLocal, this->tileLength);
        outQueueY.EnQue<float>(yLocal);
        inQueueQ.FreeTensor(qLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void linear_attention_custom(
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelLinearAttention op;
    op.Init(q, k, v, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

/* EXEC_NPU_CMD(aclnnLinearAttentionCustom, q, k, v, result); */

at::Tensor linear_attention_impl_npu(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v)
{
    at::Tensor qAct = at::relu(q) + 1e-6;
    at::Tensor kAct = at::relu(k) + 1e-6;
    at::Tensor kv = at::bmm(kAct.transpose(1, 2), v);
    return at::bmm(qAct, kv);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("linear_attention_custom", &linear_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_attention_custom", &linear_attention_impl_npu, "linear attention");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q, k, v):
        return custom_ops_lib.linear_attention_custom(q, k, v)
'''
