project_json_src='''
[
    {
        "op": "GemmGroupNormMinBiasAddCustom",
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
                "name": "group_norm_weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "group_norm_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "add_bias",
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
BEGIN_TILING_DATA_DEF(GemmGroupNormMinBiasAddCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rowCount);
TILING_DATA_FIELD_DEF(uint32_t, kDim);
TILING_DATA_FIELD_DEF(uint32_t, colCount);
TILING_DATA_FIELD_DEF(uint32_t, numGroups);
TILING_DATA_FIELD_DEF(uint32_t, channelsPerGroup);
TILING_DATA_FIELD_DEF(float, invChannelsPerGroup);
TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    GemmGroupNormMinBiasAddCustom,
    GemmGroupNormMinBiasAddCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "gemm_group_norm_min_bias_add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
inline bool IsVectorWithLength(const gert::Shape* shape, int64_t expected)
{
    return shape != nullptr && shape->GetDimNum() == 1 && shape->GetDim(0) == expected;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* weightShape = context->GetInputShape(1);
    const gert::StorageShape* gemmBiasShape = context->GetInputShape(2);
    const gert::StorageShape* gnWeightShape = context->GetInputShape(3);
    const gert::StorageShape* gnBiasShape = context->GetInputShape(4);
    const gert::StorageShape* addBiasShape = context->GetInputShape(5);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (xShape == nullptr || weightShape == nullptr || gemmBiasShape == nullptr ||
        gnWeightShape == nullptr || gnBiasShape == nullptr || addBiasShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xStorageShape = xShape->GetStorageShape();
    const auto weightStorageShape = weightShape->GetStorageShape();
    if (xStorageShape.GetDimNum() != 2 || weightStorageShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int32_t rowCount = static_cast<int32_t>(xStorageShape.GetDim(0));
    const int32_t kDim = static_cast<int32_t>(xStorageShape.GetDim(1));
    const int32_t weightK = static_cast<int32_t>(weightStorageShape.GetDim(0));
    const int32_t colCount = static_cast<int32_t>(weightStorageShape.GetDim(1));
    const int64_t* numGroupsPtr = attrs->GetAttrPointer<int64_t>(0);
    const float* epsPtr = attrs->GetAttrPointer<float>(1);
    if (rowCount <= 0 || kDim <= 0 || colCount <= 0 || kDim != weightK || numGroupsPtr == nullptr || *numGroupsPtr <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t numGroups = static_cast<uint32_t>(*numGroupsPtr);
    if (static_cast<uint32_t>(colCount) % numGroups != 0) {
        return ge::GRAPH_FAILED;
    }
    if (!IsVectorWithLength(&gemmBiasShape->GetStorageShape(), colCount) ||
        !IsVectorWithLength(&gnWeightShape->GetStorageShape(), colCount) ||
        !IsVectorWithLength(&gnBiasShape->GetStorageShape(), colCount) ||
        !IsVectorWithLength(&addBiasShape->GetStorageShape(), 1)) {
        return ge::GRAPH_FAILED;
    }

    GemmGroupNormMinBiasAddCustomTilingData tiling;
    tiling.set_rowCount(static_cast<uint32_t>(rowCount));
    tiling.set_kDim(static_cast<uint32_t>(kDim));
    tiling.set_colCount(static_cast<uint32_t>(colCount));
    tiling.set_numGroups(numGroups);
    tiling.set_channelsPerGroup(static_cast<uint32_t>(colCount) / numGroups);
    tiling.set_invChannelsPerGroup(1.0f / static_cast<float>(colCount / static_cast<int32_t>(numGroups)));
    tiling.set_eps(epsPtr == nullptr ? 1e-5f : *epsPtr);

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    if (xShape == nullptr || xShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }

    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, 1);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GemmGroupNormMinBiasAddCustom : public OpDef {
public:
    explicit GemmGroupNormMinBiasAddCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("gemm_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("group_norm_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("group_norm_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("add_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("num_groups").AttrType(REQUIRED).Int();
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-5f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(GemmGroupNormMinBiasAddCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"
#include <cfloat>
#include <cmath>

using namespace AscendC;

class GemmGroupNormMinBiasAddKernel {
public:
    __aicore__ inline GemmGroupNormMinBiasAddKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR gemmBias,
        GM_ADDR groupNormWeight,
        GM_ADDR groupNormBias,
        GM_ADDR addBias,
        GM_ADDR y,
        uint32_t rowCount,
        uint32_t kDim,
        uint32_t colCount,
        uint32_t numGroups,
        uint32_t channelsPerGroup,
        float invChannelsPerGroup,
        float eps)
    {
        this->rowCount = rowCount;
        this->kDim = kDim;
        this->colCount = colCount;
        this->numGroups = numGroups;
        this->channelsPerGroup = channelsPerGroup;
        this->invChannelsPerGroup = invChannelsPerGroup;
        this->eps = eps;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), rowCount * kDim);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight), kDim * colCount);
        gemmBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gemmBias), colCount);
        groupNormWeightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(groupNormWeight), colCount);
        groupNormBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(groupNormBias), colCount);
        addBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(addBias), 1);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), rowCount);

        pipe.InitBuffer(calcBuf, this->channelsPerGroup * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        PostProcess();
    }

private:
    __aicore__ inline float ComputeInvStd(float variance) const
    {
        return 1.0f / sqrt(variance + this->eps);
    }

    __aicore__ inline void PostProcess()
    {
        const float addBiasValue = addBiasGm.GetValue(0);
        for (uint32_t row = 0; row < this->rowCount; ++row) {
            float rowMin = FLT_MAX;
            const uint32_t rowOffset = row * this->colCount;
            for (uint32_t groupIdx = 0; groupIdx < this->numGroups; ++groupIdx) {
                LocalTensor<float> groupLocal = calcBuf.Get<float>();
                const uint32_t groupChannelStart = groupIdx * this->channelsPerGroup;
                float meanValue = 0.0f;
                for (uint32_t channelIdx = 0; channelIdx < this->channelsPerGroup; ++channelIdx) {
                    const uint32_t colIdx = groupChannelStart + channelIdx;
                    float value = gemmBiasGm.GetValue(colIdx);
                    for (uint32_t kIdx = 0; kIdx < this->kDim; ++kIdx) {
                        value += xGm.GetValue(row * this->kDim + kIdx) * weightGm.GetValue(kIdx * this->colCount + colIdx);
                    }
                    groupLocal.SetValue(channelIdx, value);
                    meanValue += value;
                }
                meanValue = meanValue * this->invChannelsPerGroup;

                float variance = 0.0f;
                for (uint32_t channelIdx = 0; channelIdx < this->channelsPerGroup; ++channelIdx) {
                    const float centered = groupLocal.GetValue(channelIdx) - meanValue;
                    variance += centered * centered;
                }
                variance = variance * this->invChannelsPerGroup;
                const float invStd = ComputeInvStd(variance);

                for (uint32_t channelIdx = 0; channelIdx < this->channelsPerGroup; ++channelIdx) {
                    const uint32_t colIdx = groupChannelStart + channelIdx;
                    const float rawValue = groupLocal.GetValue(channelIdx);
                    const float normalized = (rawValue - meanValue) * invStd;
                    const float affineValue =
                        normalized * groupNormWeightGm.GetValue(colIdx) + groupNormBiasGm.GetValue(colIdx);
                    if (affineValue < rowMin) {
                        rowMin = affineValue;
                    }
                }
            }
            yGm.SetValue(row, rowMin + addBiasValue);
        }
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> calcBuf;
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> gemmBiasGm;
    GlobalTensor<float> groupNormWeightGm;
    GlobalTensor<float> groupNormBiasGm;
    GlobalTensor<float> addBiasGm;
    GlobalTensor<float> yGm;
    uint32_t rowCount = 0;
    uint32_t kDim = 0;
    uint32_t colCount = 0;
    uint32_t numGroups = 0;
    uint32_t channelsPerGroup = 0;
    float invChannelsPerGroup = 0.0f;
    float eps = 1e-5f;
};

extern "C" __global__ __aicore__ void gemm_group_norm_min_bias_add_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR gemm_bias,
    GM_ADDR group_norm_weight,
    GM_ADDR group_norm_bias,
    GM_ADDR add_bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    GemmGroupNormMinBiasAddKernel op;
    op.Init(
        x,
        weight,
        gemm_bias,
        group_norm_weight,
        group_norm_bias,
        add_bias,
        y,
        tilingData.rowCount,
        tilingData.kDim,
        tilingData.colCount,
        tilingData.numGroups,
        tilingData.channelsPerGroup,
        tilingData.invChannelsPerGroup,
        tilingData.eps);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gemm_group_norm_min_bias_add_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& gemm_bias,
    const at::Tensor& group_norm_weight,
    const at::Tensor& group_norm_bias,
    const at::Tensor& add_bias,
    int64_t num_groups,
    double eps = 1e-5)
{
    at::Tensor result = at::empty({x.size(0), 1}, x.options());
    EXEC_NPU_CMD(
        aclnnGemmGroupNormMinBiasAddCustom,
        x,
        weight,
        gemm_bias,
        group_norm_weight,
        group_norm_bias,
        add_bias,
        num_groups,
        eps,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gemm_group_norm_min_bias_add_custom", &gemm_group_norm_min_bias_add_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gemm_group_norm_min_bias_add_custom",
        &gemm_group_norm_min_bias_add_custom_impl_npu,
        "gemm + group_norm + reduce_min + bias_add custom");
}
"""

model_src='''
import torch
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, num_groups, eps):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features, dtype=torch.float32))
        self.gemm_bias = torch.nn.Parameter(torch.randn(out_features, dtype=torch.float32))
        self.group_norm_weight = torch.nn.Parameter(torch.randn(out_features, dtype=torch.float32))
        self.group_norm_bias = torch.nn.Parameter(torch.randn(out_features, dtype=torch.float32))
        self.add_bias = torch.nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, x):
        return custom_ops_lib.gemm_group_norm_min_bias_add_custom(
            x,
            self.weight,
            self.gemm_bias,
            self.group_norm_weight,
            self.group_norm_bias,
            self.add_bias,
            self.num_groups,
            self.eps,
        )
'''
