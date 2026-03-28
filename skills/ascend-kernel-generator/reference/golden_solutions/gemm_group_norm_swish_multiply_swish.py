project_json_src='''
[
    {
        "op": "GemmGroupNormSwishMultiplySwishCustom",
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
                "name": "gemm_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "gn_weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "gn_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "multiply_weight",
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
                "name": "num_groups",
                "param_type": "required",
                "type": "int"
            },
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
BEGIN_TILING_DATA_DEF(GemmGroupNormSwishMultiplySwishCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, mDim);
TILING_DATA_FIELD_DEF(uint32_t, nDim);
TILING_DATA_FIELD_DEF(uint32_t, kDim);
TILING_DATA_FIELD_DEF(uint32_t, numGroups);
TILING_DATA_FIELD_DEF(uint32_t, groupSize);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, invGroupSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    GemmGroupNormSwishMultiplySwishCustom,
    GemmGroupNormSwishMultiplySwishCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "gemm_group_norm_swish_multiply_swish_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto shapeX = context->GetInputTensor(0)->GetOriginShape();
    auto shapeWeight = context->GetInputTensor(1)->GetOriginShape();
    auto shapeBias = context->GetInputTensor(2)->GetOriginShape();
    auto shapeGnWeight = context->GetInputTensor(3)->GetOriginShape();
    auto shapeGnBias = context->GetInputTensor(4)->GetOriginShape();
    auto shapeMultiply = context->GetInputTensor(5)->GetOriginShape();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();

    if (shapeX.GetDimNum() != 2 || shapeWeight.GetDimNum() != 2 || shapeBias.GetDimNum() != 1 ||
        shapeGnWeight.GetDimNum() != 1 || shapeGnBias.GetDimNum() != 1 || shapeMultiply.GetDimNum() != 1 ||
        attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int32_t m = shapeX.GetDim(0);
    const int32_t k = shapeX.GetDim(1);
    const int32_t weightK = shapeWeight.GetDim(0);
    const int32_t n = shapeWeight.GetDim(1);
    const int32_t biasN = shapeBias.GetDim(0);
    const int32_t gnWeightN = shapeGnWeight.GetDim(0);
    const int32_t gnBiasN = shapeGnBias.GetDim(0);
    const int32_t multiplyN = shapeMultiply.GetDim(0);
    const int64_t *numGroupsPtr = attrs->GetAttrPointer<int64_t>(0);
    const float *epsPtr = attrs->GetAttrPointer<float>(1);
    if (m <= 0 || k <= 0 || n <= 0 || k != weightK || biasN != n || gnWeightN != n || gnBiasN != n ||
        multiplyN != n || numGroupsPtr == nullptr || *numGroupsPtr <= 0 || n % (*numGroupsPtr) != 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t numGroups = static_cast<uint32_t>(*numGroupsPtr);
    const uint32_t groupSize = static_cast<uint32_t>(n / static_cast<int32_t>(numGroups));
    const uint32_t blockDim = static_cast<uint32_t>(m >= 8 ? 8 : m);

    GemmGroupNormSwishMultiplySwishCustomTilingData tiling;
    tiling.set_mDim(static_cast<uint32_t>(m));
    tiling.set_nDim(static_cast<uint32_t>(n));
    tiling.set_kDim(static_cast<uint32_t>(k));
    tiling.set_numGroups(numGroups);
    tiling.set_groupSize(groupSize);
    tiling.set_blockDim(blockDim);
    tiling.set_epsilon(epsPtr == nullptr ? 1e-5f : *epsPtr);
    tiling.set_invGroupSize(1.0f / static_cast<float>(groupSize));

    context->SetBlockDim(blockDim);
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
    const gert::Shape *gnWeightShape = context->GetInputShape(3);
    const gert::Shape *gnBiasShape = context->GetInputShape(4);
    const gert::Shape *multiplyShape = context->GetInputShape(5);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr || gnWeightShape == nullptr ||
        gnBiasShape == nullptr || multiplyShape == nullptr || attrs == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2 || biasShape->GetDimNum() != 1 ||
        gnWeightShape->GetDimNum() != 1 || gnBiasShape->GetDimNum() != 1 || multiplyShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    const int64_t *numGroupsPtr = attrs->GetAttrPointer<int64_t>(0);
    if (numGroupsPtr == nullptr || *numGroupsPtr <= 0) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != weightShape->GetDim(0) || weightShape->GetDim(1) != biasShape->GetDim(0) ||
        weightShape->GetDim(1) != gnWeightShape->GetDim(0) || weightShape->GetDim(1) != gnBiasShape->GetDim(0) ||
        weightShape->GetDim(1) != multiplyShape->GetDim(0) || weightShape->GetDim(1) % (*numGroupsPtr) != 0) {
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
class GemmGroupNormSwishMultiplySwishCustom : public OpDef {
public:
    explicit GemmGroupNormSwishMultiplySwishCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("gemm_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("gn_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("gn_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("multiply_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("num_groups").AttrType(REQUIRED).Int();
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-5f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(GemmGroupNormSwishMultiplySwishCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class GemmGroupNormSwishMultiplySwishKernel {
public:
    __aicore__ inline GemmGroupNormSwishMultiplySwishKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR gemmBias,
        GM_ADDR gnWeight,
        GM_ADDR gnBias,
        GM_ADDR multiplyWeight,
        GM_ADDR y,
        uint32_t mDim,
        uint32_t nDim,
        uint32_t kDim,
        uint32_t numGroups,
        uint32_t groupSize,
        uint32_t blockDim,
        float epsilon,
        float invGroupSize)
    {
        this->rowCount = mDim;
        this->colCount = nDim;
        this->reduceCount = kDim;
        this->numGroups = numGroups;
        this->groupSize = groupSize;
        this->blockDim = blockDim;
        this->epsilon = epsilon;
        this->invGroupSize = invGroupSize;

        xGlobal.SetGlobalBuffer((__gm__ float *)x, static_cast<uint64_t>(mDim) * kDim);
        weightGlobal.SetGlobalBuffer((__gm__ float *)weight, static_cast<uint64_t>(kDim) * nDim);
        yGlobal.SetGlobalBuffer((__gm__ float *)y, static_cast<uint64_t>(mDim) * nDim);
        gemmBiasGlobal.SetGlobalBuffer((__gm__ float *)gemmBias, nDim);
        gnWeightGlobal.SetGlobalBuffer((__gm__ float *)gnWeight, nDim);
        gnBiasGlobal.SetGlobalBuffer((__gm__ float *)gnBias, nDim);
        multiplyWeightGlobal.SetGlobalBuffer((__gm__ float *)multiplyWeight, nDim);

        pipe.InitBuffer(groupBuf, groupSize * sizeof(float));
        pipe.InitBuffer(scalarBuf, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = GetBlockIdx();
        LocalTensor<float> groupLocal = groupBuf.Get<float>();
        LocalTensor<float> scalarLocal = scalarBuf.Get<float>();

        for (uint32_t row = blockIdx; row < this->rowCount; row += this->blockDim) {
            const uint64_t xRowOffset = static_cast<uint64_t>(row) * this->reduceCount;
            const uint64_t yRowOffset = static_cast<uint64_t>(row) * this->colCount;

            for (uint32_t group = 0; group < this->numGroups; ++group) {
                const uint32_t groupOffset = group * this->groupSize;
                float meanValue = 0.0f;
                for (uint32_t idx = 0; idx < this->groupSize; ++idx) {
                    const uint32_t col = groupOffset + idx;
                    float accum = gemmBiasGlobal.GetValue(col);
                    for (uint32_t kIdx = 0; kIdx < this->reduceCount; ++kIdx) {
                        const float xValue = xGlobal.GetValue(xRowOffset + kIdx);
                        const float weightValue =
                            weightGlobal.GetValue(static_cast<uint64_t>(kIdx) * this->colCount + col);
                        accum += xValue * weightValue;
                    }
                    groupLocal.SetValue(idx, accum);
                    meanValue += accum;
                }
                meanValue *= this->invGroupSize;

                float varianceValue = 0.0f;
                for (uint32_t idx = 0; idx < this->groupSize; ++idx) {
                    const float centered = groupLocal.GetValue(idx) - meanValue;
                    varianceValue += centered * centered;
                }

                scalarLocal.SetValue(0, varianceValue * this->invGroupSize + this->epsilon);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                Sqrt(scalarLocal, scalarLocal, 1);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const float invStdValue = 1.0f / scalarLocal.GetValue(0);

                for (uint32_t idx = 0; idx < this->groupSize; ++idx) {
                    const uint32_t col = groupOffset + idx;
                    float value = groupLocal.GetValue(idx);
                    value = (value - meanValue) * invStdValue;
                    value = value * gnWeightGlobal.GetValue(col) + gnBiasGlobal.GetValue(col);
                    value = value * Sigmoid(value);
                    value = value * multiplyWeightGlobal.GetValue(col);
                    value = value * Sigmoid(value);
                    yGlobal.SetValue(yRowOffset + col, value);
                }
            }
        }
    }

private:
    __aicore__ inline float Sigmoid(float value) const
    {
        if (value >= 0.0f) {
            const float expNeg = FastExp(-value);
            return 1.0f / (1.0f + expNeg);
        }
        const float expPos = FastExp(value);
        return expPos / (1.0f + expPos);
    }

    __aicore__ inline float FastExp(float x) const
    {
        const float kLn2 = 0.69314718056f;
        if (x < -20.0f) {
            return 0.0f;
        }
        int32_t k = 0;
        while (x > 0.5f * kLn2) {
            x -= kLn2;
            ++k;
        }
        while (x < -0.5f * kLn2) {
            x += kLn2;
            --k;
        }

        const float x2 = x * x;
        const float x3 = x2 * x;
        const float x4 = x3 * x;
        const float x5 = x4 * x;
        float result = 1.0f + x + 0.5f * x2 + 0.16666667f * x3 + 0.04166667f * x4 + 0.0083333333f * x5;
        if (k > 0) {
            for (int32_t i = 0; i < k; ++i) {
                result *= 2.0f;
            }
        } else {
            for (int32_t i = 0; i < -k; ++i) {
                result *= 0.5f;
            }
        }
        return result;
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> groupBuf;
    TBuf<TPosition::VECCALC> scalarBuf;
    GlobalTensor<float> xGlobal;
    GlobalTensor<float> weightGlobal;
    GlobalTensor<float> yGlobal;
    GlobalTensor<float> gemmBiasGlobal;
    GlobalTensor<float> gnWeightGlobal;
    GlobalTensor<float> gnBiasGlobal;
    GlobalTensor<float> multiplyWeightGlobal;
    uint32_t rowCount = 0;
    uint32_t colCount = 0;
    uint32_t reduceCount = 0;
    uint32_t numGroups = 1;
    uint32_t groupSize = 1;
    uint32_t blockDim = 1;
    float epsilon = 1e-5f;
    float invGroupSize = 1.0f;
};

extern "C" __global__ __aicore__ void gemm_group_norm_swish_multiply_swish_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR gemm_bias,
    GM_ADDR gn_weight,
    GM_ADDR gn_bias,
    GM_ADDR multiply_weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    GemmGroupNormSwishMultiplySwishKernel op;
    op.Init(
        x,
        weight,
        gemm_bias,
        gn_weight,
        gn_bias,
        multiply_weight,
        y,
        tilingData.mDim,
        tilingData.nDim,
        tilingData.kDim,
        tilingData.numGroups,
        tilingData.groupSize,
        tilingData.blockDim,
        tilingData.epsilon,
        tilingData.invGroupSize);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gemm_group_norm_swish_multiply_swish_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &gemm_bias,
    const at::Tensor &gn_weight,
    const at::Tensor &gn_bias,
    const at::Tensor &multiply_weight,
    int64_t num_groups,
    double eps = 1e-5)
{
    at::Tensor result = at::empty({x.size(0), weight.size(1)}, x.options());
    EXEC_NPU_CMD(
        aclnnGemmGroupNormSwishMultiplySwishCustom,
        x,
        weight,
        gemm_bias,
        gn_weight,
        gn_bias,
        multiply_weight,
        num_groups,
        eps,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "gemm_group_norm_swish_multiply_swish_custom",
        &gemm_group_norm_swish_multiply_swish_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gemm_group_norm_swish_multiply_swish_custom",
        &gemm_group_norm_swish_multiply_swish_custom_impl_npu,
        "gemm + group_norm + swish + multiply + swish");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = torch.nn.Linear(in_features, out_features)
        self.group_norm = torch.nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = torch.nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        weight = self.gemm.weight.transpose(0, 1).contiguous()
        return custom_ops_lib.gemm_group_norm_swish_multiply_swish_custom(
            x,
            weight,
            self.gemm.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )
'''
