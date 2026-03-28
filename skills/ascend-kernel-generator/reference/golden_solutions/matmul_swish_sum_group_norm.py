project_json_src='''
[
    {
        "op": "MatmulSwishSumGroupNormCustom",
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
                "name": "matmul_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "add_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "gamma",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "beta",
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
BEGIN_TILING_DATA_DEF(MatmulSwishSumGroupNormCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, inFeatures);
TILING_DATA_FIELD_DEF(uint32_t, outFeatures);
TILING_DATA_FIELD_DEF(uint32_t, numGroups);
TILING_DATA_FIELD_DEF(uint32_t, channelsPerGroup);
TILING_DATA_FIELD_DEF(float, invChannelsPerGroup);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulSwishSumGroupNormCustom, MatmulSwishSumGroupNormCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "matmul_swish_sum_group_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
inline bool IsVectorWithLength(const gert::Shape *shape, int64_t expected)
{
    return shape != nullptr && shape->GetDimNum() == 1 && shape->GetDim(0) == expected;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *matmulBiasShape = context->GetInputShape(2);
    const gert::StorageShape *addBiasShape = context->GetInputShape(3);
    const gert::StorageShape *gammaShape = context->GetInputShape(4);
    const gert::StorageShape *betaShape = context->GetInputShape(5);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (xShape == nullptr || weightShape == nullptr || matmulBiasShape == nullptr || addBiasShape == nullptr ||
        gammaShape == nullptr || betaShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xStorageShape = xShape->GetStorageShape();
    const auto weightStorageShape = weightShape->GetStorageShape();
    if (xStorageShape.GetDimNum() != 2 || weightStorageShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int32_t batchSize = static_cast<int32_t>(xStorageShape.GetDim(0));
    const int32_t inFeatures = static_cast<int32_t>(xStorageShape.GetDim(1));
    const int32_t weightInFeatures = static_cast<int32_t>(weightStorageShape.GetDim(0));
    const int32_t outFeatures = static_cast<int32_t>(weightStorageShape.GetDim(1));
    const int64_t *numGroupsPtr = attrs->GetAttrPointer<int64_t>(0);
    const float *epsPtr = attrs->GetAttrPointer<float>(1);
    if (batchSize <= 0 || inFeatures <= 0 || outFeatures <= 0 || weightInFeatures != inFeatures ||
        numGroupsPtr == nullptr || *numGroupsPtr <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t numGroups = static_cast<uint32_t>(*numGroupsPtr);
    if (static_cast<uint32_t>(outFeatures) % numGroups != 0) {
        return ge::GRAPH_FAILED;
    }
    if (!IsVectorWithLength(&matmulBiasShape->GetStorageShape(), outFeatures) ||
        !IsVectorWithLength(&addBiasShape->GetStorageShape(), outFeatures) ||
        !IsVectorWithLength(&gammaShape->GetStorageShape(), outFeatures) ||
        !IsVectorWithLength(&betaShape->GetStorageShape(), outFeatures)) {
        return ge::GRAPH_FAILED;
    }

    MatmulSwishSumGroupNormCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(batchSize));
    tiling.set_inFeatures(static_cast<uint32_t>(inFeatures));
    tiling.set_outFeatures(static_cast<uint32_t>(outFeatures));
    tiling.set_numGroups(numGroups);
    tiling.set_channelsPerGroup(static_cast<uint32_t>(outFeatures) / numGroups);
    tiling.set_invChannelsPerGroup(1.0f / static_cast<float>(outFeatures / static_cast<int32_t>(numGroups)));
    tiling.set_eps(epsPtr == nullptr ? 1.0e-5f : *epsPtr);
    tiling.set_blockDim(batchSize < 8 ? static_cast<uint32_t>(batchSize) : 8U);

    context->SetBlockDim(tiling.get_blockDim());
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
    const gert::Shape *matmulBiasShape = context->GetInputShape(2);
    const gert::Shape *addBiasShape = context->GetInputShape(3);
    const gert::Shape *gammaShape = context->GetInputShape(4);
    const gert::Shape *betaShape = context->GetInputShape(5);
    if (xShape == nullptr || weightShape == nullptr || matmulBiasShape == nullptr || addBiasShape == nullptr ||
        gammaShape == nullptr || betaShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2 || matmulBiasShape->GetDimNum() != 1 ||
        addBiasShape->GetDimNum() != 1 || gammaShape->GetDimNum() != 1 || betaShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != weightShape->GetDim(0)) {
        return GRAPH_FAILED;
    }
    const int64_t outFeatures = weightShape->GetDim(1);
    if (matmulBiasShape->GetDim(0) != outFeatures || addBiasShape->GetDim(0) != outFeatures ||
        gammaShape->GetDim(0) != outFeatures || betaShape->GetDim(0) != outFeatures) {
        return GRAPH_FAILED;
    }

    gert::Shape *yShape = context->GetOutputShape(0);
    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, outFeatures);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class MatmulSwishSumGroupNormCustom : public OpDef {
public:
    explicit MatmulSwishSumGroupNormCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("matmul_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("add_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("num_groups").AttrType(REQUIRED).Int();
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-5f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(MatmulSwishSumGroupNormCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class MatmulSwishSumGroupNormKernel {
public:
    __aicore__ inline MatmulSwishSumGroupNormKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR matmulBias,
        GM_ADDR addBias,
        GM_ADDR gamma,
        GM_ADDR beta,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inFeatures,
        uint32_t outFeatures,
        uint32_t numGroups,
        uint32_t channelsPerGroup,
        float invChannelsPerGroup,
        float eps,
        uint32_t blockDim)
    {
        this->batchSize = batchSize;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->numGroups = numGroups;
        this->channelsPerGroup = channelsPerGroup;
        this->invChannelsPerGroup = invChannelsPerGroup;
        this->eps = eps;
        this->blockDim = blockDim;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), static_cast<uint64_t>(batchSize) * inFeatures);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight), static_cast<uint64_t>(inFeatures) * outFeatures);
        matmulBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matmulBias), outFeatures);
        addBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(addBias), outFeatures);
        gammaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gamma), outFeatures);
        betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(beta), outFeatures);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), static_cast<uint64_t>(batchSize) * outFeatures);

        pipe.InitBuffer(groupBuf, channelsPerGroup * sizeof(float));
        pipe.InitBuffer(scalarBuf, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t row = GetBlockIdx(); row < this->batchSize; row += this->blockDim) {
            const uint64_t xBase = static_cast<uint64_t>(row) * this->inFeatures;
            const uint64_t yBase = static_cast<uint64_t>(row) * this->outFeatures;
            for (uint32_t groupIdx = 0; groupIdx < this->numGroups; ++groupIdx) {
                ComputeGroup(row, xBase, yBase, groupIdx);
            }
        }
    }

private:
    __aicore__ inline float FastExp(float x) const
    {
        constexpr float ln2 = 0.69314718056f;
        if (x < -20.0f) {
            return 0.0f;
        }
        if (x > 20.0f) {
            x = 20.0f;
        }

        int32_t k = 0;
        while (x > 0.5f * ln2) {
            x -= ln2;
            ++k;
        }
        while (x < -0.5f * ln2) {
            x += ln2;
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

    __aicore__ inline float Sigmoid(float x) const
    {
        return 1.0f / (1.0f + FastExp(-x));
    }

    __aicore__ inline float ComputeInvStd(LocalTensor<float> &scalarLocal, float variance) const
    {
        scalarLocal.SetValue(0, variance + this->eps);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Sqrt(scalarLocal, scalarLocal, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return 1.0f / scalarLocal.GetValue(0);
    }

    __aicore__ inline void ComputeGroup(
        uint32_t row,
        uint64_t xBase,
        uint64_t yBase,
        uint32_t groupIdx)
    {
        LocalTensor<float> groupLocal = groupBuf.Get<float>();
        LocalTensor<float> scalarLocal = scalarBuf.Get<float>();
        const uint32_t groupStart = groupIdx * this->channelsPerGroup;

        float meanValue = 0.0f;
        for (uint32_t channelIdx = 0; channelIdx < this->channelsPerGroup; ++channelIdx) {
            const uint32_t outCol = groupStart + channelIdx;
            float acc = matmulBiasGm.GetValue(outCol);
            for (uint32_t inIdx = 0; inIdx < this->inFeatures; ++inIdx) {
                acc += xGm.GetValue(xBase + inIdx) *
                    weightGm.GetValue(static_cast<uint64_t>(inIdx) * this->outFeatures + outCol);
            }
            const float swishValue = acc * Sigmoid(acc);
            const float summedValue = swishValue + addBiasGm.GetValue(outCol);
            groupLocal.SetValue(channelIdx, summedValue);
            meanValue += summedValue;
        }
        meanValue *= this->invChannelsPerGroup;

        float varianceValue = 0.0f;
        for (uint32_t channelIdx = 0; channelIdx < this->channelsPerGroup; ++channelIdx) {
            const float centered = groupLocal.GetValue(channelIdx) - meanValue;
            varianceValue += centered * centered;
        }
        varianceValue *= this->invChannelsPerGroup;
        const float invStd = ComputeInvStd(scalarLocal, varianceValue);

        for (uint32_t channelIdx = 0; channelIdx < this->channelsPerGroup; ++channelIdx) {
            const uint32_t outCol = groupStart + channelIdx;
            const float normalized = (groupLocal.GetValue(channelIdx) - meanValue) * invStd;
            const float outputValue = normalized * gammaGm.GetValue(outCol) + betaGm.GetValue(outCol);
            yGm.SetValue(yBase + outCol, outputValue);
        }
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> groupBuf;
    TBuf<TPosition::VECCALC> scalarBuf;
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> matmulBiasGm;
    GlobalTensor<float> addBiasGm;
    GlobalTensor<float> gammaGm;
    GlobalTensor<float> betaGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
    uint32_t inFeatures = 0;
    uint32_t outFeatures = 0;
    uint32_t numGroups = 0;
    uint32_t channelsPerGroup = 0;
    float invChannelsPerGroup = 0.0f;
    float eps = 1.0e-5f;
    uint32_t blockDim = 1;
};

extern "C" __global__ __aicore__ void matmul_swish_sum_group_norm_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR matmul_bias,
    GM_ADDR add_bias,
    GM_ADDR gamma,
    GM_ADDR beta,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    MatmulSwishSumGroupNormKernel op;
    op.Init(
        x,
        weight,
        matmul_bias,
        add_bias,
        gamma,
        beta,
        y,
        tilingData.batchSize,
        tilingData.inFeatures,
        tilingData.outFeatures,
        tilingData.numGroups,
        tilingData.channelsPerGroup,
        tilingData.invChannelsPerGroup,
        tilingData.eps,
        tilingData.blockDim);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_swish_sum_group_norm_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &matmul_bias,
    const at::Tensor &add_bias,
    const at::Tensor &gamma,
    const at::Tensor &beta,
    int64_t num_groups,
    double eps = 1e-5)
{
    at::Tensor result = at::empty({x.size(0), weight.size(1)}, x.options());
    EXEC_NPU_CMD(
        aclnnMatmulSwishSumGroupNormCustom,
        x,
        weight,
        matmul_bias,
        add_bias,
        gamma,
        beta,
        num_groups,
        eps,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_swish_sum_group_norm_custom", &matmul_swish_sum_group_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "matmul_swish_sum_group_norm_custom",
        &matmul_swish_sum_group_norm_custom_impl_npu,
        "matmul + swish + add + group_norm custom");
}
"""

model_src='''
import torch
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, num_groups, eps=1e-5):
        super().__init__()
        self.num_groups = int(num_groups)
        self.eps = float(eps)
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float32))
        self.matmul_bias = torch.nn.Parameter(torch.randn(out_features, dtype=torch.float32))
        self.add_bias = torch.nn.Parameter(torch.randn(out_features, dtype=torch.float32))
        self.gamma = torch.nn.Parameter(torch.randn(out_features, dtype=torch.float32))
        self.beta = torch.nn.Parameter(torch.randn(out_features, dtype=torch.float32))

    def forward(self, x):
        return custom_ops_lib.matmul_swish_sum_group_norm_custom(
            x,
            self.weight,
            self.matmul_bias,
            self.add_bias,
            self.gamma,
            self.beta,
            self.num_groups,
            self.eps,
        )
'''
