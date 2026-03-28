project_json_src='''
[
    {
        "op": "KvChatBatchAttnCustom",
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
                "name": "k_cache",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "v_cache",
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
BEGIN_TILING_DATA_DEF(KvChatBatchAttnCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, qLen);
TILING_DATA_FIELD_DEF(uint32_t, kvLen);
TILING_DATA_FIELD_DEF(uint32_t, dModel);
TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvChatBatchAttnCustom, KvChatBatchAttnCustomTilingData)
}
"""

host_operator_src="""
#include <cmath>
#include "kv_chat_batch_attn_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto qShape = context->GetInputTensor(0)->GetOriginShape();
    auto kShape = context->GetInputTensor(1)->GetOriginShape();
    auto vShape = context->GetInputTensor(2)->GetOriginShape();
    if (qShape.GetDimNum() != 3 || kShape.GetDimNum() != 3 || vShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t qLen = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t dModel = static_cast<uint32_t>(qShape.GetDim(2));
    const uint32_t kBatch = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t kvLen = static_cast<uint32_t>(kShape.GetDim(1));
    const uint32_t kDim = static_cast<uint32_t>(kShape.GetDim(2));
    const uint32_t vBatch = static_cast<uint32_t>(vShape.GetDim(0));
    const uint32_t vLen = static_cast<uint32_t>(vShape.GetDim(1));
    const uint32_t vDim = static_cast<uint32_t>(vShape.GetDim(2));

    if (batchSize == 0 || qLen != 1 || kBatch != batchSize || vBatch != batchSize) {
        return ge::GRAPH_FAILED;
    }
    if (kvLen == 0 || dModel == 0 || kDim != dModel || vLen != kvLen || vDim != dModel) {
        return ge::GRAPH_FAILED;
    }

    KvChatBatchAttnCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_qLen(qLen);
    tiling.set_kvLen(kvLen);
    tiling.set_dModel(dModel);
    tiling.set_scale(1.0f / std::sqrt(static_cast<float>(dModel)));

    context->SetBlockDim(batchSize);
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
class KvChatBatchAttnCustom : public OpDef {
public:
    explicit KvChatBatchAttnCustom(const char *name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("k_cache").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("v_cache").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(KvChatBatchAttnCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

class KernelKvChatBatchAttn {
public:
    __aicore__ inline KernelKvChatBatchAttn() {}

    __aicore__ inline void Init(
        GM_ADDR q,
        GM_ADDR kCache,
        GM_ADDR vCache,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t qLen,
        uint32_t kvLen,
        uint32_t dModel,
        float scale)
    {
        this->batchSize = batchSize;
        this->qLen = qLen;
        this->kvLen = kvLen;
        this->dModel = dModel;
        this->scale = scale;

        const uint32_t batchIdx = AscendC::GetBlockIdx();
        const uint32_t qBatchOffset = batchIdx * qLen * dModel;
        const uint32_t kvBatchOffset = batchIdx * kvLen * dModel;

        qGm.SetGlobalBuffer((__gm__ float *)q + qBatchOffset, qLen * dModel);
        kGm.SetGlobalBuffer((__gm__ float *)kCache + kvBatchOffset, kvLen * dModel);
        vGm.SetGlobalBuffer((__gm__ float *)vCache + kvBatchOffset, kvLen * dModel);
        yGm.SetGlobalBuffer((__gm__ float *)y + qBatchOffset, qLen * dModel);

        pipe.InitBuffer(qBuf, dModel * sizeof(float));
        pipe.InitBuffer(scoreBuf, kvLen * sizeof(float));
        pipe.InitBuffer(tempBuf, kvLen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> qLocal = qBuf.Get<float>();
        AscendC::DataCopy(qLocal, qGm, dModel);

        AscendC::LocalTensor<float> scores = scoreBuf.Get<float>();
        for (uint32_t kvIdx = 0; kvIdx < kvLen; ++kvIdx) {
            float acc = 0.0f;
            const uint32_t base = kvIdx * dModel;
            for (uint32_t col = 0; col < dModel; ++col) {
                acc += qLocal.GetValue(col) * kGm.GetValue(base + col);
            }
            scores.SetValue(kvIdx, acc * scale);
        }

        float maxScore = scores.GetValue(0);
        for (uint32_t kvIdx = 1; kvIdx < kvLen; ++kvIdx) {
            const float score = scores.GetValue(kvIdx);
            if (score > maxScore) {
                maxScore = score;
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

        AscendC::LocalTensor<float> expScores = tempBuf.Get<float>();
        AscendC::Adds(expScores, scores, -maxScore, kvLen);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(expScores, expScores, kvLen);

        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

        float denom = 0.0f;
        for (uint32_t kvIdx = 0; kvIdx < kvLen; ++kvIdx) {
            denom += expScores.GetValue(kvIdx);
        }
        const float invDenom = denom == 0.0f ? 0.0f : 1.0f / denom;
        for (uint32_t kvIdx = 0; kvIdx < kvLen; ++kvIdx) {
            scores.SetValue(kvIdx, expScores.GetValue(kvIdx) * invDenom);
        }

        for (uint32_t col = 0; col < dModel; ++col) {
            float outValue = 0.0f;
            for (uint32_t kvIdx = 0; kvIdx < kvLen; ++kvIdx) {
                outValue += scores.GetValue(kvIdx) * vGm.GetValue(kvIdx * dModel + col);
            }
            yGm.SetValue(col, outValue);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scoreBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf;
    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t qLen;
    uint32_t kvLen;
    uint32_t dModel;
    float scale;
};

extern "C" __global__ __aicore__ void kv_chat_batch_attn_custom(
    GM_ADDR q,
    GM_ADDR k_cache,
    GM_ADDR v_cache,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelKvChatBatchAttn op;
    op.Init(
        q,
        k_cache,
        v_cache,
        y,
        tiling_data.batchSize,
        tiling_data.qLen,
        tiling_data.kvLen,
        tiling_data.dModel,
        tiling_data.scale);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor kv_cached_chat_batch_attention_impl_npu(
    const at::Tensor &q,
    const at::Tensor &k_cache,
    const at::Tensor &v_cache)
{
    at::Tensor result = at::empty_like(q);
    EXEC_NPU_CMD(aclnnKvChatBatchAttnCustom, q, k_cache, v_cache, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("kv_chat_batch_attn_custom", &kv_cached_chat_batch_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "kv_chat_batch_attn_custom",
        &kv_cached_chat_batch_attention_impl_npu,
        "kv cached chat batch attention");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q, k_cache, v_cache):
        return custom_ops_lib.kv_chat_batch_attn_custom(q, k_cache, v_cache)
'''
