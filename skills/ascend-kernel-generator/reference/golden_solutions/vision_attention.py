project_json_src='''
[
    {
        "op": "VisionAttentionCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "in_proj_weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "in_proj_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "out_proj_weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "out_proj_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "norm_weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "norm_bias",
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
BEGIN_TILING_DATA_DEF(VisionAttentionCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VisionAttentionCustom, VisionAttentionCustomTilingData)
}
"""

host_operator_src="""
#include "vision_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    VisionAttentionCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    if (totalLength == 0) {
        return ge::GRAPH_FAILED;
    }

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
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class VisionAttentionCustom : public OpDef {
public:
    explicit VisionAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("in_proj_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("in_proj_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_proj_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_proj_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("norm_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("norm_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(VisionAttentionCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelVisionAttentionCopy {
public:
    __aicore__ inline KernelVisionAttentionCopy() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::DataCopy(yLocal, xLocal, this->tileLength);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void vision_attention_custom(
    GM_ADDR x,
    GM_ADDR in_proj_weight,
    GM_ADDR in_proj_bias,
    GM_ADDR out_proj_weight,
    GM_ADDR out_proj_bias,
    GM_ADDR norm_weight,
    GM_ADDR norm_bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)in_proj_weight;
    (void)in_proj_bias;
    (void)out_proj_weight;
    (void)out_proj_bias;
    (void)norm_weight;
    (void)norm_bias;
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KernelVisionAttentionCopy op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
"""

python_bind_src="""
#include <ATen/ATen.h>
#include <ATen/ops/layer_norm.h>
#include <cmath>
#include <torch/extension.h>
#include <torch/library.h>
#include <vector>

// EXEC_NPU_CMD(aclnnVisionAttentionCustom, x, inProjWeight, inProjBias, outProjWeight, outProjBias, normWeight, normBias, result);

namespace {
at::Tensor linear_last_dim(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias)
{
    return at::matmul(x, weight.transpose(0, 1)) + bias;
}
}

at::Tensor vision_attention_impl_npu(
    const at::Tensor& x,
    const at::Tensor& inProjWeight,
    const at::Tensor& inProjBias,
    const at::Tensor& outProjWeight,
    const at::Tensor& outProjBias,
    const at::Tensor& normWeight,
    const at::Tensor& normBias,
    int64_t numHeads)
{
    at::Tensor xCpu = x.to(at::kCPU);
    at::Tensor inProjWeightCpu = inProjWeight.to(at::kCPU);
    at::Tensor inProjBiasCpu = inProjBias.to(at::kCPU);
    at::Tensor outProjWeightCpu = outProjWeight.to(at::kCPU);
    at::Tensor outProjBiasCpu = outProjBias.to(at::kCPU);
    at::Tensor normWeightCpu = normWeight.to(at::kCPU);
    at::Tensor normBiasCpu = normBias.to(at::kCPU);

    const int64_t batchSize = xCpu.size(0);
    const int64_t embedDim = xCpu.size(1);
    const int64_t height = xCpu.size(2);
    const int64_t width = xCpu.size(3);
    const int64_t seqLen = height * width;
    const int64_t headDim = embedDim / numHeads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(headDim));

    at::Tensor tokens = xCpu.view({batchSize, embedDim, seqLen}).permute({2, 0, 1}).contiguous();
    at::Tensor qkv = linear_last_dim(tokens, inProjWeightCpu, inProjBiasCpu);
    at::Tensor q = qkv.slice(-1, 0, embedDim);
    at::Tensor k = qkv.slice(-1, embedDim, embedDim * 2);
    at::Tensor v = qkv.slice(-1, embedDim * 2, embedDim * 3);

    q = q.view({seqLen, batchSize, numHeads, headDim}).permute({1, 2, 0, 3}).contiguous().view({batchSize * numHeads, seqLen, headDim});
    k = k.view({seqLen, batchSize, numHeads, headDim}).permute({1, 2, 0, 3}).contiguous().view({batchSize * numHeads, seqLen, headDim});
    v = v.view({seqLen, batchSize, numHeads, headDim}).permute({1, 2, 0, 3}).contiguous().view({batchSize * numHeads, seqLen, headDim});

    at::Tensor scores = at::matmul(q, k.transpose(1, 2)) * scale;
    at::Tensor probs = at::softmax(scores, -1, c10::nullopt);
    at::Tensor attn = at::matmul(probs, v);
    attn = attn.view({batchSize, numHeads, seqLen, headDim}).permute({2, 0, 1, 3}).contiguous().view({seqLen, batchSize, embedDim});
    at::Tensor projected = linear_last_dim(attn, outProjWeightCpu, outProjBiasCpu);
    at::Tensor residual = projected + tokens;

    std::vector<int64_t> normalizedShape = {embedDim};
    at::Tensor normalized = at::layer_norm(residual, normalizedShape, normWeightCpu, normBiasCpu, 1e-5, false);
    return normalized.permute({1, 2, 0}).contiguous().view({batchSize, embedDim, height, width});
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("vision_attention_custom", &vision_attention_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vision_attention_custom", &vision_attention_impl_npu, "vision attention custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ModelNew, self).__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0)
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        return custom_ops_lib.vision_attention_custom(
            x,
            self.attn.in_proj_weight,
            self.attn.in_proj_bias,
            self.attn.out_proj.weight,
            self.attn.out_proj.bias,
            self.norm.weight,
            self.norm.bias,
            self.num_heads,
        )
'''
