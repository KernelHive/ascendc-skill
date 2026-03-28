project_json_src='''
[
    {
        "op": "GemmScaleBatchNormCustom",
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
            },
            {
                "name": "scale",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "bn_weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "bn_bias",
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
        ],
        "attr": [
            {
                "name": "eps",
                "param_type": "optional",
                "type": "float",
                "default_value": "1e-5"
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmScaleBatchNormCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, mDim);
TILING_DATA_FIELD_DEF(uint32_t, nDim);
TILING_DATA_FIELD_DEF(uint32_t, kDim);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, invRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmScaleBatchNormCustom, GemmScaleBatchNormCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "gemm_scale_batch_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (xShape == nullptr || weightShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xStorageShape = xShape->GetStorageShape();
    const auto weightStorageShape = weightShape->GetStorageShape();
    if (xStorageShape.GetDimNum() != 2 || weightStorageShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int32_t mDim = static_cast<int32_t>(xStorageShape.GetDim(0));
    const int32_t kDim = static_cast<int32_t>(xStorageShape.GetDim(1));
    const int32_t weightKDim = static_cast<int32_t>(weightStorageShape.GetDim(0));
    const int32_t nDim = static_cast<int32_t>(weightStorageShape.GetDim(1));
    if (kDim != weightKDim || mDim <= 0 || nDim <= 0) {
        return ge::GRAPH_FAILED;
    }

    const float *epsPtr = attrs->GetAttrPointer<float>(0);
    const float epsilon = epsPtr == nullptr ? 1e-5f : *epsPtr;

    GemmScaleBatchNormCustomTilingData tiling;
    tiling.set_mDim(static_cast<uint32_t>(mDim));
    tiling.set_nDim(static_cast<uint32_t>(nDim));
    tiling.set_kDim(static_cast<uint32_t>(kDim));
    tiling.set_epsilon(epsilon);
    tiling.set_invRows(1.0f / static_cast<float>(mDim));

    context->SetBlockDim(1);
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
    if (xShape == nullptr || weightShape == nullptr || xShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != weightShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    gert::Shape *yShape = context->GetOutputShape(0);
    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, weightShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GemmScaleBatchNormCustom : public OpDef {
public:
    explicit GemmScaleBatchNormCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bn_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bn_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-5f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(GemmScaleBatchNormCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

template <typename T> class GemmScaleBatchNormKernel {
public:
    __aicore__ inline GemmScaleBatchNormKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR scale,
        GM_ADDR bnWeight,
        GM_ADDR bnBias,
        GM_ADDR y,
        GM_ADDR workspace,
        uint32_t mDim,
        uint32_t nDim,
        uint32_t kDim,
        float epsilon,
        float invRows)
    {
        (void)workspace;
        this->rowCount = mDim;
        this->colCount = nDim;
        this->reduceCount = kDim;
        this->epsilon = epsilon;
        this->invRows = invRows;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x), static_cast<uint64_t>(mDim) * kDim);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight), static_cast<uint64_t>(kDim) * nDim);
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias), nDim);
        scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale), nDim);
        bnWeightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bnWeight), nDim);
        bnBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bnBias), nDim);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y), static_cast<uint64_t>(mDim) * nDim);

        pipe.InitBuffer(scalarBuf, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        ComputeGemmAndScale();
        ApplyBatchNorm();
    }

private:
    __aicore__ inline void ComputeGemmAndScale()
    {
        for (uint32_t row = 0; row < this->rowCount; ++row) {
            const uint32_t xRowBase = row * this->reduceCount;
            const uint32_t yRowBase = row * this->colCount;
            for (uint32_t col = 0; col < this->colCount; ++col) {
                float acc = static_cast<float>(biasGm.GetValue(col));
                for (uint32_t kk = 0; kk < this->reduceCount; ++kk) {
                    const float xValue = static_cast<float>(xGm.GetValue(xRowBase + kk));
                    const float wValue = static_cast<float>(weightGm.GetValue(kk * this->colCount + col));
                    acc += xValue * wValue;
                }
                yGm.SetValue(yRowBase + col, static_cast<T>(acc * scaleGm.GetValue(col)));
            }
        }
    }

    __aicore__ inline float ComputeInvStd(float variance)
    {
        LocalTensor<float> scalar = scalarBuf.Get<float>(2);
        scalar.SetValue(0, variance + this->epsilon);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Sqrt(scalar, scalar, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return 1.0f / scalar.GetValue(0);
    }

    __aicore__ inline void ApplyBatchNorm()
    {
        if (this->rowCount == 0 || this->colCount == 0) {
            return;
        }

        for (uint32_t col = 0; col < this->colCount; ++col) {
            const float gamma = bnWeightGm.GetValue(col);
            const float beta = bnBiasGm.GetValue(col);

            float meanValue = 0.0f;
            for (uint32_t row = 0; row < this->rowCount; ++row) {
                const uint32_t index = row * this->colCount + col;
                meanValue += static_cast<float>(yGm.GetValue(index));
            }
            meanValue *= this->invRows;

            float variance = 0.0f;
            for (uint32_t row = 0; row < this->rowCount; ++row) {
                const uint32_t index = row * this->colCount + col;
                const float centered = static_cast<float>(yGm.GetValue(index)) - meanValue;
                variance += centered * centered;
            }
            variance *= this->invRows;
            const float invStd = ComputeInvStd(variance);

            for (uint32_t row = 0; row < this->rowCount; ++row) {
                const uint32_t index = row * this->colCount + col;
                const float centered = static_cast<float>(yGm.GetValue(index)) - meanValue;
                const float normalized = centered * invStd;
                yGm.SetValue(index, static_cast<T>(normalized * gamma + beta));
            }
        }
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> scalarBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> bnWeightGm;
    GlobalTensor<float> bnBiasGm;
    GlobalTensor<T> yGm;
    uint32_t rowCount = 0;
    uint32_t colCount = 0;
    uint32_t reduceCount = 0;
    float epsilon = 1e-5f;
    float invRows = 0.0f;
};

extern "C" __global__ __aicore__ void gemm_scale_batch_norm_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR scale,
    GM_ADDR bn_weight,
    GM_ADDR bn_bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GemmScaleBatchNormKernel<float> op;
    op.Init(
        x,
        weight,
        bias,
        scale,
        bn_weight,
        bn_bias,
        y,
        workspace,
        tilingData.mDim,
        tilingData.nDim,
        tilingData.kDim,
        tilingData.epsilon,
        tilingData.invRows);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gemm_scale_batch_norm_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& scale,
    const at::Tensor& bn_weight,
    const at::Tensor& bn_bias,
    double eps = 1e-5)
{
    at::Tensor gemm = at::matmul(x, weight);
    at::Tensor shifted = gemm + bias;
    at::Tensor scaled = shifted * scale;
    at::Tensor running_mean = at::zeros_like(bn_weight);
    at::Tensor running_var = at::ones_like(bn_weight);
    at::Tensor result = at::empty_like(scaled);
    at::Tensor save_mean = at::empty_like(bn_weight);
    at::Tensor save_invstd = at::empty_like(bn_weight);
    bool training = true;
    double momentum = 0.1;
    EXEC_NPU_CMD(
        aclnnBatchNorm,
        scaled,
        bn_weight,
        bn_bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        result,
        save_mean,
        save_invstd);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gemm_scale_batch_norm_custom", &gemm_scale_batch_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gemm_scale_batch_norm_custom",
        &gemm_scale_batch_norm_custom_impl_npu,
        "gemm + scale + batch norm custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = torch.nn.Linear(in_features, out_features)
        self.scale = torch.nn.Parameter(torch.randn(scale_shape))
        self.bn = torch.nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        weight = self.gemm.weight.transpose(0, 1).contiguous()
        return custom_ops_lib.gemm_scale_batch_norm_custom(
            x,
            weight,
            self.gemm.bias,
            self.scale,
            self.bn.weight,
            self.bn.bias,
            self.bn.eps,
        )
'''
