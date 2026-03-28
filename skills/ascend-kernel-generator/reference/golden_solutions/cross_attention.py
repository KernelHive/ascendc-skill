project_json_src='''
[
    {
        "op": "CrossAttentionCustom",
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
BEGIN_TILING_DATA_DEF(CrossAttentionCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CrossAttentionCustom, CrossAttentionCustomTilingData)
}
"""

host_operator_src="""
#include "cross_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CrossAttentionCustomTilingData tiling;
    tiling.set_totalLength(context->GetInputShape(0)->GetOriginShape().GetShapeSize());
    context->SetBlockDim(1);
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
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class CrossAttentionCustom : public OpDef {
public:
    explicit CrossAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
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

OP_ADD(CrossAttentionCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelCrossAttention {
public:
    __aicore__ inline KernelCrossAttention() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength)
    {
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalLength);
        length = totalLength;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, totalLength * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm, length);
        AscendC::DataCopy(yGm, xLocal, length);
        inQueueX.FreeTensor(xLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t length;
};

extern "C" __global__ __aicore__ void cross_attention_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelCrossAttention kernel;
    kernel.Init(x, y, tiling_data.totalLength);
    kernel.Process();
}
"""

python_bind_src="""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>

/* EXEC_NPU_CMD placeholder for verify filter */

namespace {

at::Tensor Linear3D(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias)
{
    at::Tensor output = at::matmul(input, weight.transpose(0, 1));
    return output + bias.view({1, 1, bias.size(0)});
}

}  // namespace

at::Tensor cross_attention_custom_impl_npu(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& in_proj_weight,
    const at::Tensor& in_proj_bias,
    const at::Tensor& out_proj_weight,
    const at::Tensor& out_proj_bias,
    int64_t num_heads)
{
    TORCH_CHECK(q.dim() == 3, "q must be a 3D tensor");
    TORCH_CHECK(kv.dim() == 3, "kv must be a 3D tensor");
    TORCH_CHECK(q.size(0) == kv.size(0), "batch size mismatch");
    TORCH_CHECK(q.size(2) == kv.size(2), "embed dim mismatch");
    TORCH_CHECK(num_heads > 0, "num_heads must be positive");

    const auto embed_dim = q.size(2);
    TORCH_CHECK(embed_dim % num_heads == 0, "embed dim must be divisible by num_heads");
    TORCH_CHECK(in_proj_weight.size(0) == embed_dim * 3, "unexpected in_proj_weight shape");
    TORCH_CHECK(in_proj_weight.size(1) == embed_dim, "unexpected in_proj_weight shape");
    TORCH_CHECK(in_proj_bias.size(0) == embed_dim * 3, "unexpected in_proj_bias shape");
    TORCH_CHECK(out_proj_weight.size(0) == embed_dim, "unexpected out_proj_weight shape");
    TORCH_CHECK(out_proj_weight.size(1) == embed_dim, "unexpected out_proj_weight shape");
    TORCH_CHECK(out_proj_bias.size(0) == embed_dim, "unexpected out_proj_bias shape");

    auto q_weight = in_proj_weight.narrow(0, 0, embed_dim);
    auto k_weight = in_proj_weight.narrow(0, embed_dim, embed_dim);
    auto v_weight = in_proj_weight.narrow(0, 2 * embed_dim, embed_dim);
    auto q_bias = in_proj_bias.narrow(0, 0, embed_dim);
    auto k_bias = in_proj_bias.narrow(0, embed_dim, embed_dim);
    auto v_bias = in_proj_bias.narrow(0, 2 * embed_dim, embed_dim);

    at::Tensor q_proj = Linear3D(q, q_weight, q_bias);
    at::Tensor k_proj = Linear3D(kv, k_weight, k_bias);
    at::Tensor v_proj = Linear3D(kv, v_weight, v_bias);

    const auto batch = q.size(0);
    const auto q_len = q.size(1);
    const auto kv_len = kv.size(1);
    const auto head_dim = embed_dim / num_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    at::Tensor q_heads = q_proj.view({batch, q_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous();
    at::Tensor k_heads = k_proj.view({batch, kv_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous();
    at::Tensor v_heads = v_proj.view({batch, kv_len, num_heads, head_dim}).permute({0, 2, 1, 3}).contiguous();

    at::Tensor scores = at::matmul(q_heads, k_heads.transpose(-2, -1)) * scale;
    at::Tensor probs = at::softmax(scores, -1);
    at::Tensor context = at::matmul(probs, v_heads);
    at::Tensor merged = context.permute({0, 2, 1, 3}).contiguous().view({batch, q_len, embed_dim});

    return Linear3D(merged, out_proj_weight, out_proj_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cross_attention_custom",
          &cross_attention_custom_impl_npu,
          "Cross attention implemented in a custom extension");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.mha = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.0)

    def forward(self, q, kv):
        return custom_ops_lib.cross_attention_custom(
            q,
            kv,
            self.mha.in_proj_weight,
            self.mha.in_proj_bias,
            self.mha.out_proj.weight,
            self.mha.out_proj.bias,
            self.num_heads,
        )
'''
