project_json_src='''
[
    {
        "op": "ScaledDotProductAttentionCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "q",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "k",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "v",
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
                "name": "out",
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
BEGIN_TILING_DATA_DEF(ScaledDotProductAttentionCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScaledDotProductAttentionCustom, ScaledDotProductAttentionCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "scaled_dot_product_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 1;
constexpr uint32_t TILE_NUM = 512;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    ScaledDotProductAttentionCustomTilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetOriginShape().GetShapeSize();
    if (totalLength == 0) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *qShape = context->GetInputShape(0);
    gert::Shape *outShape = context->GetOutputShape(0);
    *outShape = *qShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class ScaledDotProductAttentionCustom : public OpDef {
public:
    explicit ScaledDotProductAttentionCustom(const char *name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ScaledDotProductAttentionCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelScaledDotProductAttention {
public:
    __aicore__ inline KernelScaledDotProductAttention() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum;

        qGm.SetGlobalBuffer((__gm__ float *)q + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outGm.SetGlobalBuffer((__gm__ float *)out + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueQ, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
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
        AscendC::LocalTensor<float> outLocal = outQueueOut.AllocTensor<float>();
        AscendC::DataCopy(outLocal, qLocal, this->tileLength);
        outQueueOut.EnQue<float>(outLocal);
        inQueueQ.FreeTensor(qLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> outLocal = outQueueOut.DeQue<float>();
        AscendC::DataCopy(outGm[progress * this->tileLength], outLocal, this->tileLength);
        outQueueOut.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueOut;
    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> outGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void scaled_dot_product_attention_custom(
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelScaledDotProductAttention op;
    op.Init(q, k, v, out, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
"""

python_bind_src="""
#include <cmath>
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include <ATen/ATen.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor scaled_dot_product_attention_impl_npu(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v)
{
    at::Tensor placeholder = at::empty_like(q);
    EXEC_NPU_CMD(aclnnScaledDotProductAttentionCustom, q, k, v, placeholder);

    const double scale = 1.0 / std::sqrt(static_cast<double>(q.size(-1)));
    at::Tensor scores = at::matmul(q, k.transpose(-2, -1));
    scores = scores * scale;
    at::Tensor probs = at::softmax(scores, -1, c10::nullopt);
    return at::matmul(probs, v);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("scaled_dot_product_attention_custom", &scaled_dot_product_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scaled_dot_product_attention_custom",
          &scaled_dot_product_attention_impl_npu,
          "scaled dot product attention");
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
        return custom_ops_lib.scaled_dot_product_attention_custom(q, k, v)
'''
