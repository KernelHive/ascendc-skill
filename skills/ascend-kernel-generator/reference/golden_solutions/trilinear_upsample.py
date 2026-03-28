project_json_src='''
[
    {
        "op": "TrilinearUpsampleCustom",
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
BEGIN_TILING_DATA_DEF(TrilinearUpsampleCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TrilinearUpsampleCustom, TrilinearUpsampleCustomTilingData)
}
"""

host_operator_src="""
#include "trilinear_upsample_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t kBlockDim = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    if (shape.GetDim(0) != 2 || shape.GetDim(1) != 16 || shape.GetDim(2) != 32 ||
        shape.GetDim(3) != 32 || shape.GetDim(4) != 32) {
        return ge::GRAPH_FAILED;
    }

    TrilinearUpsampleCustomTilingData tiling;
    tiling.set_size(2 * 16 * 32 * 32 * 32);
    context->SetBlockDim(kBlockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr || inputShape->GetDimNum() != 5) {
        return GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, 2);
    outputShape->SetDim(1, 16);
    outputShape->SetDim(2, 64);
    outputShape->SetDim(3, 64);
    outputShape->SetDim(4, 64);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class TrilinearUpsampleCustom : public OpDef {
public:
    explicit TrilinearUpsampleCustom(const char* name) : OpDef(name)
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

OP_ADD(TrilinearUpsampleCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

__aicore__ constexpr int32_t kAxisSize = 64;
__aicore__ constexpr int32_t kInputSize = 32;
__aicore__ constexpr float kInvDenom = 1.0f / 63.0f;

__aicore__ constexpr int32_t kLowerIndex[kAxisSize] = {
    0, 0, 0, 1, 1, 2, 2, 3,
    3, 4, 4, 5, 5, 6, 6, 7,
    7, 8, 8, 9, 9, 10, 10, 11,
    11, 12, 12, 13, 13, 14, 14, 15,
    15, 16, 16, 17, 17, 18, 18, 19,
    19, 20, 20, 21, 21, 22, 22, 23,
    23, 24, 24, 25, 25, 26, 26, 27,
    27, 28, 28, 29, 29, 30, 30, 31
};

__aicore__ constexpr int32_t kUpperIndex[kAxisSize] = {
    0, 1, 1, 2, 2, 3, 3, 4,
    4, 5, 5, 6, 6, 7, 7, 8,
    8, 9, 9, 10, 10, 11, 11, 12,
    12, 13, 13, 14, 14, 15, 15, 16,
    16, 17, 17, 18, 18, 19, 19, 20,
    20, 21, 21, 22, 22, 23, 23, 24,
    24, 25, 25, 26, 26, 27, 27, 28,
    28, 29, 29, 30, 30, 31, 31, 31
};

__aicore__ constexpr int32_t kNumerator[kAxisSize] = {
    0, 31, 62, 30, 61, 29, 60, 28,
    59, 27, 58, 26, 57, 25, 56, 24,
    55, 23, 54, 22, 53, 21, 52, 20,
    51, 19, 50, 18, 49, 17, 48, 16,
    47, 15, 46, 14, 45, 13, 44, 12,
    43, 11, 42, 10, 41, 9, 40, 8,
    39, 7, 38, 6, 37, 5, 36, 4,
    35, 3, 34, 2, 33, 1, 32, 0
};

class KernelTrilinearUpsample {
public:
    __aicore__ inline KernelTrilinearUpsample() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x, kBatch * kChannels * kInputPlane);
        yGm.SetGlobalBuffer((__gm__ float*)y, kBatch * kChannels * kOutputPlane);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        for (uint32_t planeIdx = blockIdx; planeIdx < kTotalPlanes; planeIdx += kBlockDim) {
            ProcessPlane(planeIdx);
        }
    }

private:
    __aicore__ inline void ProcessPlane(uint32_t planeIdx)
    {
        const uint32_t ncIdx = planeIdx / kOutDepth;
        const uint32_t outZ = planeIdx % kOutDepth;
        const uint32_t inputBase = ncIdx * kInputPlane;
        const uint32_t outputBase = ncIdx * kOutputPlane + outZ * kOutputDepthStride;

        const int32_t z0 = kLowerIndex[outZ];
        const int32_t z1 = kUpperIndex[outZ];
        const float wz1 = static_cast<float>(kNumerator[outZ]) * kInvDenom;
        const float wz0 = 1.0f - wz1;

        for (uint32_t outY = 0; outY < kOutHeight; ++outY) {
            const int32_t y0 = kLowerIndex[outY];
            const int32_t y1 = kUpperIndex[outY];
            const float wy1 = static_cast<float>(kNumerator[outY]) * kInvDenom;
            const float wy0 = 1.0f - wy1;
            const uint32_t outputRowBase = outputBase + outY * kOutWidth;

            for (uint32_t outX = 0; outX < kOutWidth; ++outX) {
                const int32_t x0 = kLowerIndex[outX];
                const int32_t x1 = kUpperIndex[outX];
                const float wx1 = static_cast<float>(kNumerator[outX]) * kInvDenom;
                const float wx0 = 1.0f - wx1;

                const uint32_t z0Base = inputBase + static_cast<uint32_t>(z0) * kInputDepthStride;
                const uint32_t z1Base = inputBase + static_cast<uint32_t>(z1) * kInputDepthStride;
                const uint32_t y00Base = z0Base + static_cast<uint32_t>(y0) * kInputRowStride;
                const uint32_t y01Base = z0Base + static_cast<uint32_t>(y1) * kInputRowStride;
                const uint32_t y10Base = z1Base + static_cast<uint32_t>(y0) * kInputRowStride;
                const uint32_t y11Base = z1Base + static_cast<uint32_t>(y1) * kInputRowStride;

                const float c000 = xGm.GetValue(y00Base + static_cast<uint32_t>(x0));
                const float c001 = xGm.GetValue(y00Base + static_cast<uint32_t>(x1));
                const float c010 = xGm.GetValue(y01Base + static_cast<uint32_t>(x0));
                const float c011 = xGm.GetValue(y01Base + static_cast<uint32_t>(x1));
                const float c100 = xGm.GetValue(y10Base + static_cast<uint32_t>(x0));
                const float c101 = xGm.GetValue(y10Base + static_cast<uint32_t>(x1));
                const float c110 = xGm.GetValue(y11Base + static_cast<uint32_t>(x0));
                const float c111 = xGm.GetValue(y11Base + static_cast<uint32_t>(x1));

                const float c00 = c000 * wx0 + c001 * wx1;
                const float c01 = c010 * wx0 + c011 * wx1;
                const float c10 = c100 * wx0 + c101 * wx1;
                const float c11 = c110 * wx0 + c111 * wx1;
                const float c0 = c00 * wy0 + c01 * wy1;
                const float c1 = c10 * wy0 + c11 * wy1;
                yGm.SetValue(outputRowBase + outX, c0 * wz0 + c1 * wz1);
            }
        }
    }

private:
    static constexpr uint32_t kBatch = 2;
    static constexpr uint32_t kChannels = 16;
    static constexpr uint32_t kInDepth = 32;
    static constexpr uint32_t kInHeight = 32;
    static constexpr uint32_t kInWidth = 32;
    static constexpr uint32_t kOutDepth = 64;
    static constexpr uint32_t kOutHeight = 64;
    static constexpr uint32_t kOutWidth = 64;
    static constexpr uint32_t kBlockDim = 8;
    static constexpr uint32_t kInputRowStride = kInWidth;
    static constexpr uint32_t kInputDepthStride = kInHeight * kInWidth;
    static constexpr uint32_t kOutputDepthStride = kOutHeight * kOutWidth;
    static constexpr uint32_t kInputPlane = kInDepth * kInHeight * kInWidth;
    static constexpr uint32_t kOutputPlane = kOutDepth * kOutHeight * kOutWidth;
    static constexpr uint32_t kTotalPlanes = kBatch * kChannels * kOutDepth;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
};

extern "C" __global__ __aicore__ void trilinear_upsample_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    (void)tilingData;
    KernelTrilinearUpsample op;
    op.Init(x, y);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor trilinear_upsample_custom_impl_npu(const at::Tensor& self)
{
    auto outputShape = self.sizes().vec();
    outputShape[2] *= 2;
    outputShape[3] *= 2;
    outputShape[4] *= 2;
    at::Tensor result = at::empty(outputShape, self.options());
    EXEC_NPU_CMD(aclnnTrilinearUpsampleCustom, self, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("trilinear_upsample_custom", &trilinear_upsample_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trilinear_upsample_custom", &trilinear_upsample_custom_impl_npu, "trilinear_upsample_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.trilinear_upsample_custom(x)
'''
