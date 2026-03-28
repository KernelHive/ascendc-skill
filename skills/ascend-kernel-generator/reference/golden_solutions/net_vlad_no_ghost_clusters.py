project_json_src='''
[
    {
        "op": "NetVladNoGhostClustersCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "clusters",
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
            },
            {
                "name": "clusters2",
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
                "name": "bn_eps",
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
BEGIN_TILING_DATA_DEF(NetVladNoGhostClustersCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, sampleCount);
TILING_DATA_FIELD_DEF(uint32_t, featureSize);
TILING_DATA_FIELD_DEF(uint32_t, clusterSize);
TILING_DATA_FIELD_DEF(uint32_t, flattenedOutputSize);
TILING_DATA_FIELD_DEF(float, bnEps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NetVladNoGhostClustersCustom, NetVladNoGhostClustersCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "net_vlad_no_ghost_clusters_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(0);
    const gert::StorageShape *clustersShape = context->GetInputShape(1);
    const gert::StorageShape *bnWeightShape = context->GetInputShape(2);
    const gert::StorageShape *bnBiasShape = context->GetInputShape(3);
    const gert::StorageShape *clusters2Shape = context->GetInputShape(4);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (xShape == nullptr || clustersShape == nullptr || bnWeightShape == nullptr || bnBiasShape == nullptr ||
        clusters2Shape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xStorageShape = xShape->GetStorageShape();
    const auto clustersStorageShape = clustersShape->GetStorageShape();
    const auto bnWeightStorageShape = bnWeightShape->GetStorageShape();
    const auto bnBiasStorageShape = bnBiasShape->GetStorageShape();
    const auto clusters2StorageShape = clusters2Shape->GetStorageShape();
    if (xStorageShape.GetDimNum() != 3 || clustersStorageShape.GetDimNum() != 2 || bnWeightStorageShape.GetDimNum() != 1 ||
        bnBiasStorageShape.GetDimNum() != 1 || clusters2StorageShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xStorageShape.GetDim(0));
    const uint32_t sampleCount = static_cast<uint32_t>(xStorageShape.GetDim(1));
    const uint32_t featureSize = static_cast<uint32_t>(xStorageShape.GetDim(2));
    const uint32_t clusterSize = static_cast<uint32_t>(clustersStorageShape.GetDim(1));
    if (batchSize == 0 || sampleCount == 0 || featureSize == 0 || clusterSize == 0) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint32_t>(clustersStorageShape.GetDim(0)) != featureSize ||
        static_cast<uint32_t>(bnWeightStorageShape.GetDim(0)) != clusterSize ||
        static_cast<uint32_t>(bnBiasStorageShape.GetDim(0)) != clusterSize ||
        static_cast<uint32_t>(clusters2StorageShape.GetDim(0)) != 1 ||
        static_cast<uint32_t>(clusters2StorageShape.GetDim(1)) != featureSize ||
        static_cast<uint32_t>(clusters2StorageShape.GetDim(2)) != clusterSize) {
        return ge::GRAPH_FAILED;
    }

    const float *bnEpsPtr = attrs->GetAttrPointer<float>(0);
    NetVladNoGhostClustersCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_sampleCount(sampleCount);
    tiling.set_featureSize(featureSize);
    tiling.set_clusterSize(clusterSize);
    tiling.set_flattenedOutputSize(featureSize * clusterSize);
    tiling.set_bnEps(bnEpsPtr == nullptr ? 1.0e-5f : *bnEpsPtr);

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
    const gert::Shape *clustersShape = context->GetInputShape(1);
    const gert::Shape *bnWeightShape = context->GetInputShape(2);
    const gert::Shape *bnBiasShape = context->GetInputShape(3);
    const gert::Shape *clusters2Shape = context->GetInputShape(4);
    if (xShape == nullptr || clustersShape == nullptr || bnWeightShape == nullptr || bnBiasShape == nullptr ||
        clusters2Shape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 3 || clustersShape->GetDimNum() != 2 || bnWeightShape->GetDimNum() != 1 ||
        bnBiasShape->GetDimNum() != 1 || clusters2Shape->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(2) != clustersShape->GetDim(0) ||
        clustersShape->GetDim(1) != bnWeightShape->GetDim(0) ||
        bnWeightShape->GetDim(0) != bnBiasShape->GetDim(0) ||
        clusters2Shape->GetDim(0) != 1 ||
        clusters2Shape->GetDim(1) != xShape->GetDim(2) ||
        clusters2Shape->GetDim(2) != clustersShape->GetDim(1)) {
        return GRAPH_FAILED;
    }

    gert::Shape *yShape = context->GetOutputShape(0);
    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, xShape->GetDim(2) * clustersShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class NetVladNoGhostClustersCustom : public OpDef {
public:
    explicit NetVladNoGhostClustersCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("clusters").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bn_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bn_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("clusters2").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("bn_eps").AttrType(OPTIONAL).Float(1.0e-5f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(NetVladNoGhostClustersCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class NetVladNoGhostClustersKernel {
public:
    __aicore__ inline NetVladNoGhostClustersKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR clusters,
        GM_ADDR bnWeight,
        GM_ADDR bnBias,
        GM_ADDR clusters2,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t sampleCount,
        uint32_t featureSize,
        uint32_t clusterSize,
        uint32_t flattenedOutputSize,
        float bnEps)
    {
        this->batchSize = batchSize;
        this->sampleCount = sampleCount;
        this->featureSize = featureSize;
        this->clusterSize = clusterSize;
        this->flattenedOutputSize = flattenedOutputSize;
        this->bnEps = bnEps;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), static_cast<uint64_t>(batchSize) * sampleCount * featureSize);
        clustersGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(clusters), static_cast<uint64_t>(featureSize) * clusterSize);
        bnWeightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bnWeight), clusterSize);
        bnBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bnBias), clusterSize);
        clusters2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(clusters2), static_cast<uint64_t>(featureSize) * clusterSize);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), static_cast<uint64_t>(batchSize) * flattenedOutputSize);

        pipe.InitBuffer(statsBuf, clusterSize * 2U * sizeof(float));
        pipe.InitBuffer(logitsBuf, clusterSize * sizeof(float));
        pipe.InitBuffer(assignBuf, clusterSize * sizeof(float));
        pipe.InitBuffer(outputBuf, flattenedOutputSize * sizeof(float));
        pipe.InitBuffer(scalarBuf, 8U * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> statsLocal = statsBuf.Get<float>();
        LocalTensor<float> meanLocal = statsLocal;
        LocalTensor<float> invStdLocal = statsLocal[clusterSize];
        ComputeBatchNormStats(meanLocal, invStdLocal);

        for (uint32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            ComputeBatch(batchIdx, meanLocal, invStdLocal);
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

    __aicore__ inline float SqrtScalar(float value)
    {
        LocalTensor<float> scalarLocal = scalarBuf.Get<float>();
        scalarLocal.SetValue(0, value);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Sqrt(scalarLocal, scalarLocal, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return scalarLocal.GetValue(0);
    }

    __aicore__ inline float ComputeAssignmentLogit(uint64_t xBase, uint32_t clusterIdx) const
    {
        float acc = 0.0f;
        for (uint32_t featureIdx = 0; featureIdx < featureSize; ++featureIdx) {
            acc += xGm.GetValue(xBase + featureIdx) *
                clustersGm.GetValue(static_cast<uint64_t>(featureIdx) * clusterSize + clusterIdx);
        }
        return acc;
    }

    __aicore__ inline void ComputeBatchNormStats(LocalTensor<float> &meanLocal, LocalTensor<float> &invStdLocal)
    {
        const float invTotalSamples = (1.0f / batchSize) / sampleCount;

        for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
            meanLocal.SetValue(clusterIdx, 0.0f);
            invStdLocal.SetValue(clusterIdx, 0.0f);
        }

        for (uint32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            for (uint32_t sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
                const uint64_t xBase = (static_cast<uint64_t>(batchIdx) * sampleCount + sampleIdx) * featureSize;
                for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
                    const float logit = ComputeAssignmentLogit(xBase, clusterIdx);
                    meanLocal.SetValue(clusterIdx, meanLocal.GetValue(clusterIdx) + logit);
                }
            }
        }
        for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
            meanLocal.SetValue(clusterIdx, meanLocal.GetValue(clusterIdx) * invTotalSamples);
        }

        for (uint32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            for (uint32_t sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
                const uint64_t xBase = (static_cast<uint64_t>(batchIdx) * sampleCount + sampleIdx) * featureSize;
                for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
                    const float logit = ComputeAssignmentLogit(xBase, clusterIdx);
                    const float centered = logit - meanLocal.GetValue(clusterIdx);
                    invStdLocal.SetValue(clusterIdx, invStdLocal.GetValue(clusterIdx) + centered * centered);
                }
            }
        }
        for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
            const float variance = invStdLocal.GetValue(clusterIdx) * invTotalSamples;
            const float denom = SqrtScalar(variance + bnEps);
            invStdLocal.SetValue(clusterIdx, 1.0f / denom);
        }
    }

    __aicore__ inline void ZeroBatchOutput(LocalTensor<float> &outputLocal, LocalTensor<float> &assignLocal)
    {
        for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
            assignLocal.SetValue(clusterIdx, 0.0f);
        }
        for (uint32_t outIdx = 0; outIdx < flattenedOutputSize; ++outIdx) {
            outputLocal.SetValue(outIdx, 0.0f);
        }
    }

    __aicore__ inline void SoftmaxToAssignments(
        LocalTensor<float> &logitsLocal,
        LocalTensor<float> &meanLocal,
        LocalTensor<float> &invStdLocal)
    {
        float maxValue = 0.0f;
        for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
            const float normalized = (logitsLocal.GetValue(clusterIdx) - meanLocal.GetValue(clusterIdx)) *
                invStdLocal.GetValue(clusterIdx);
            const float affine = normalized * bnWeightGm.GetValue(clusterIdx) + bnBiasGm.GetValue(clusterIdx);
            logitsLocal.SetValue(clusterIdx, affine);
            if (clusterIdx == 0 || affine > maxValue) {
                maxValue = affine;
            }
        }

        float sumExp = 0.0f;
        for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
            const float expValue = FastExp(logitsLocal.GetValue(clusterIdx) - maxValue);
            logitsLocal.SetValue(clusterIdx, expValue);
            sumExp += expValue;
        }

        const float invSum = 1.0f / sumExp;
        for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
            logitsLocal.SetValue(clusterIdx, logitsLocal.GetValue(clusterIdx) * invSum);
        }
    }

    __aicore__ inline void ComputeBatch(
        uint32_t batchIdx,
        LocalTensor<float> &meanLocal,
        LocalTensor<float> &invStdLocal)
    {
        LocalTensor<float> logitsLocal = logitsBuf.Get<float>();
        LocalTensor<float> assignLocal = assignBuf.Get<float>();
        LocalTensor<float> outputLocal = outputBuf.Get<float>();
        ZeroBatchOutput(outputLocal, assignLocal);

        for (uint32_t sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
            const uint64_t xBase = (static_cast<uint64_t>(batchIdx) * sampleCount + sampleIdx) * featureSize;
            for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
                logitsLocal.SetValue(clusterIdx, ComputeAssignmentLogit(xBase, clusterIdx));
            }
            SoftmaxToAssignments(logitsLocal, meanLocal, invStdLocal);

            for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
                assignLocal.SetValue(clusterIdx, assignLocal.GetValue(clusterIdx) + logitsLocal.GetValue(clusterIdx));
            }

            for (uint32_t featureIdx = 0; featureIdx < featureSize; ++featureIdx) {
                const float xValue = xGm.GetValue(xBase + featureIdx);
                const uint32_t featureBase = featureIdx * clusterSize;
                for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
                    const uint32_t outIdx = featureBase + clusterIdx;
                    outputLocal.SetValue(outIdx, outputLocal.GetValue(outIdx) + logitsLocal.GetValue(clusterIdx) * xValue);
                }
            }
        }

        for (uint32_t featureIdx = 0; featureIdx < featureSize; ++featureIdx) {
            const uint32_t featureBase = featureIdx * clusterSize;
            const uint64_t clusters2Base = static_cast<uint64_t>(featureIdx) * clusterSize;
            for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
                const uint32_t outIdx = featureBase + clusterIdx;
                const float residual = assignLocal.GetValue(clusterIdx) * clusters2Gm.GetValue(clusters2Base + clusterIdx);
                outputLocal.SetValue(outIdx, outputLocal.GetValue(outIdx) - residual);
            }
        }

        constexpr float normEps = 1.0e-12f;
        for (uint32_t clusterIdx = 0; clusterIdx < clusterSize; ++clusterIdx) {
            float squaredNorm = 0.0f;
            for (uint32_t featureIdx = 0; featureIdx < featureSize; ++featureIdx) {
                const float value = outputLocal.GetValue(featureIdx * clusterSize + clusterIdx);
                squaredNorm += value * value;
            }
            float denom = SqrtScalar(squaredNorm);
            if (denom < normEps) {
                denom = normEps;
            }
            const float invNorm = 1.0f / denom;
            for (uint32_t featureIdx = 0; featureIdx < featureSize; ++featureIdx) {
                const uint32_t outIdx = featureIdx * clusterSize + clusterIdx;
                outputLocal.SetValue(outIdx, outputLocal.GetValue(outIdx) * invNorm);
            }
        }

        float flatSquaredNorm = 0.0f;
        for (uint32_t outIdx = 0; outIdx < flattenedOutputSize; ++outIdx) {
            const float value = outputLocal.GetValue(outIdx);
            flatSquaredNorm += value * value;
        }
        float flatDenom = SqrtScalar(flatSquaredNorm);
        if (flatDenom < normEps) {
            flatDenom = normEps;
        }
        const float flatInvNorm = 1.0f / flatDenom;

        const uint64_t yBase = static_cast<uint64_t>(batchIdx) * flattenedOutputSize;
        for (uint32_t outIdx = 0; outIdx < flattenedOutputSize; ++outIdx) {
            yGm.SetValue(yBase + outIdx, outputLocal.GetValue(outIdx) * flatInvNorm);
        }
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> statsBuf;
    TBuf<TPosition::VECCALC> logitsBuf;
    TBuf<TPosition::VECCALC> assignBuf;
    TBuf<TPosition::VECCALC> outputBuf;
    TBuf<TPosition::VECCALC> scalarBuf;
    GlobalTensor<float> xGm;
    GlobalTensor<float> clustersGm;
    GlobalTensor<float> bnWeightGm;
    GlobalTensor<float> bnBiasGm;
    GlobalTensor<float> clusters2Gm;
    GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
    uint32_t sampleCount = 0;
    uint32_t featureSize = 0;
    uint32_t clusterSize = 0;
    uint32_t flattenedOutputSize = 0;
    float bnEps = 1.0e-5f;
};

extern "C" __global__ __aicore__ void net_vlad_no_ghost_clusters_custom(
    GM_ADDR x,
    GM_ADDR clusters,
    GM_ADDR bn_weight,
    GM_ADDR bn_bias,
    GM_ADDR clusters2,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    NetVladNoGhostClustersKernel op;
    op.Init(
        x,
        clusters,
        bn_weight,
        bn_bias,
        clusters2,
        y,
        tilingData.batchSize,
        tilingData.sampleCount,
        tilingData.featureSize,
        tilingData.clusterSize,
        tilingData.flattenedOutputSize,
        tilingData.bnEps);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor net_vlad_no_ghost_clusters_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &clusters,
    const at::Tensor &bn_weight,
    const at::Tensor &bn_bias,
    const at::Tensor &clusters2,
    double bn_eps = 1e-5)
{
    at::Tensor result = at::empty({x.size(0), x.size(2) * clusters.size(1)}, x.options());
    EXEC_NPU_CMD(
        aclnnNetVladNoGhostClustersCustom,
        x,
        clusters,
        bn_weight,
        bn_bias,
        clusters2,
        bn_eps,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("net_vlad_no_ghost_clusters_custom", &net_vlad_no_ghost_clusters_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "net_vlad_no_ghost_clusters_custom",
        &net_vlad_no_ghost_clusters_custom_impl_npu,
        "net vlad without ghost clusters custom");
}
"""

model_src='''
import math
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = 1 / math.sqrt(feature_size)
        clusters = cluster_size + ghost_clusters

        self.clusters = torch.nn.Parameter(init_sc * torch.randn(feature_size, clusters))
        self.batch_norm = torch.nn.BatchNorm1d(clusters)
        self.clusters2 = torch.nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        del mask
        return custom_ops_lib.net_vlad_no_ghost_clusters_custom(
            x,
            self.clusters,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.clusters2,
            self.batch_norm.eps,
        )
'''
