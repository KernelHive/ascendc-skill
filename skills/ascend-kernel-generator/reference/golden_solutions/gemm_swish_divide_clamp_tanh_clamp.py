project_json_src = '''
[
    {
        "op": "GemmSwishDivideClampTanhClampCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "bias",
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

host_tiling_src = """
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmSwishDivideClampTanhClampCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, mDim);
TILING_DATA_FIELD_DEF(uint32_t, nDim);
TILING_DATA_FIELD_DEF(uint32_t, kDim);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    GemmSwishDivideClampTanhClampCustom,
    GemmSwishDivideClampTanhClampCustomTilingData)
} // namespace optiling
"""

host_operator_src = """
#include "gemm_swish_divide_clamp_tanh_clamp_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto shapeX = context->GetInputTensor(0)->GetOriginShape();
    auto shapeWeight = context->GetInputTensor(1)->GetOriginShape();
    auto shapeBias = context->GetInputTensor(2)->GetOriginShape();

    if (shapeX.GetDimNum() != 2 || shapeWeight.GetDimNum() != 2 || shapeBias.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int32_t mDim = shapeX.GetDim(0);
    const int32_t kDim = shapeX.GetDim(1);
    const int32_t weightK = shapeWeight.GetDim(0);
    const int32_t nDim = shapeWeight.GetDim(1);
    if (mDim <= 0 || nDim <= 0 || kDim <= 0 || weightK != kDim || shapeBias.GetDim(0) != nDim) {
        return ge::GRAPH_FAILED;
    }

    GemmSwishDivideClampTanhClampCustomTilingData tiling;
    tiling.set_mDim(static_cast<uint32_t>(mDim));
    tiling.set_nDim(static_cast<uint32_t>(nDim));
    tiling.set_kDim(static_cast<uint32_t>(kDim));
    tiling.set_blockDim(static_cast<uint32_t>(mDim >= 8 ? 8 : mDim));

    context->SetBlockDim(mDim >= 8 ? 8 : mDim);
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
    const gert::Shape *xShape = context->GetInputShape(0);
    const gert::Shape *weightShape = context->GetInputShape(1);
    const gert::Shape *biasShape = context->GetInputShape(2);
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != weightShape->GetDim(0) || weightShape->GetDim(1) != biasShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    gert::Shape *outShape = context->GetOutputShape(0);
    outShape->SetDimNum(2);
    outShape->SetDim(0, xShape->GetDim(0));
    outShape->SetDim(1, weightShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GemmSwishDivideClampTanhClampCustom : public OpDef {
public:
    explicit GemmSwishDivideClampTanhClampCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(GemmSwishDivideClampTanhClampCustom);
} // namespace ops
"""

kernel_src = """
#include "kernel_operator.h"

using namespace AscendC;

class GemmSwishDivideClampTanhClampKernel {
public:
    __aicore__ inline GemmSwishDivideClampTanhClampKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t mDim,
        uint32_t nDim,
        uint32_t kDim,
        uint32_t blockDim)
    {
        this->mDim = mDim;
        this->nDim = nDim;
        this->kDim = kDim;
        this->blockDim = blockDim;

        xGm.SetGlobalBuffer((__gm__ float *)x, static_cast<uint64_t>(mDim) * kDim);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, static_cast<uint64_t>(kDim) * nDim);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, nDim);
        yGm.SetGlobalBuffer((__gm__ float *)y, static_cast<uint64_t>(mDim) * nDim);

        pipe.InitBuffer(rowBuffer, nDim * sizeof(float));
        pipe.InitBuffer(sigmoidBuffer, nDim * sizeof(float));
        pipe.InitBuffer(swishBuffer, nDim * sizeof(float));
        pipe.InitBuffer(clampBuffer, nDim * sizeof(float));
        pipe.InitBuffer(tanhBuffer, nDim * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        constexpr float reciprocalDivisor = 0.5f;
        constexpr float clampMin = -1.0f;
        constexpr float clampMax = 1.0f;

        const uint32_t coreIdx = GetBlockIdx();
        LocalTensor<float> rowLocal = rowBuffer.Get<float>();
        LocalTensor<float> sigmoidLocal = sigmoidBuffer.Get<float>();
        LocalTensor<float> swishLocal = swishBuffer.Get<float>();
        LocalTensor<float> clampLocal = clampBuffer.Get<float>();
        LocalTensor<float> tanhLocal = tanhBuffer.Get<float>();

        for (uint32_t row = coreIdx; row < this->mDim; row += this->blockDim) {
            const uint64_t xOffset = static_cast<uint64_t>(row) * this->kDim;
            const uint64_t yOffset = static_cast<uint64_t>(row) * this->nDim;

            for (uint32_t col = 0; col < this->nDim; ++col) {
                float acc = biasGm.GetValue(col);
                for (uint32_t kk = 0; kk < this->kDim; ++kk) {
                    const float xValue = xGm.GetValue(xOffset + kk);
                    const float wValue = weightGm.GetValue(static_cast<uint64_t>(kk) * this->nDim + col);
                    acc += xValue * wValue;
                }
                rowLocal.SetValue(col, acc);
            }

            Sigmoid(sigmoidLocal, rowLocal, this->nDim);
            Mul(swishLocal, rowLocal, sigmoidLocal, this->nDim);
            Muls(swishLocal, swishLocal, reciprocalDivisor, this->nDim);

            for (uint32_t col = 0; col < this->nDim; ++col) {
                float value = swishLocal.GetValue(col);
                value = value < clampMin ? clampMin : value;
                value = value > clampMax ? clampMax : value;
                clampLocal.SetValue(col, value);
            }

            Tanh(tanhLocal, clampLocal, this->nDim);

            for (uint32_t col = 0; col < this->nDim; ++col) {
                float value = tanhLocal.GetValue(col);
                value = value < clampMin ? clampMin : value;
                value = value > clampMax ? clampMax : value;
                rowLocal.SetValue(col, value);
            }

            DataCopy(yGm[yOffset], rowLocal, this->nDim);
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> rowBuffer;
    TBuf<TPosition::VECCALC> sigmoidBuffer;
    TBuf<TPosition::VECCALC> swishBuffer;
    TBuf<TPosition::VECCALC> clampBuffer;
    TBuf<TPosition::VECCALC> tanhBuffer;
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t mDim = 0;
    uint32_t nDim = 0;
    uint32_t kDim = 0;
    uint32_t blockDim = 1;
};

extern "C" __global__ __aicore__ void gemm_swish_divide_clamp_tanh_clamp_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    GemmSwishDivideClampTanhClampKernel op;
    op.Init(
        x,
        weight,
        bias,
        y,
        tiling_data.mDim,
        tiling_data.nDim,
        tiling_data.kDim,
        tiling_data.blockDim);
    op.Process();
}
"""

python_bind_src = """
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gemm_swish_divide_clamp_tanh_clamp_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias)
{
    auto outputShape = std::vector<int64_t>{x.size(0), weight.size(1)};
    at::Tensor result = at::empty(outputShape, x.options());
    EXEC_NPU_CMD(aclnnGemmSwishDivideClampTanhClampCustom, x, weight, bias, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "gemm_swish_divide_clamp_tanh_clamp_custom",
        &gemm_swish_divide_clamp_tanh_clamp_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gemm_swish_divide_clamp_tanh_clamp_custom",
        &gemm_swish_divide_clamp_tanh_clamp_custom_impl_npu,
        "gemm + swish + divide + clamp + tanh + clamp");
}
"""

model_src = '''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        weight = self.gemm.weight.transpose(0, 1).contiguous()
        return custom_ops_lib.gemm_swish_divide_clamp_tanh_clamp_custom(
            x,
            weight,
            self.gemm.bias,
        )
'''
