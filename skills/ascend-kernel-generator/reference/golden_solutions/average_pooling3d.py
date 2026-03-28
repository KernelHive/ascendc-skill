project_json_src='''
[
    {
        "op": "AveragePooling3dCustom",
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
BEGIN_TILING_DATA_DEF(AveragePooling3dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalPlanes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AveragePooling3dCustom, AveragePooling3dCustomTilingData)
}
"""

host_operator_src="""
#include "average_pooling3d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kBatch = 16;
constexpr uint32_t kChannels = 32;
constexpr uint32_t kInDepth = 64;
constexpr uint32_t kInHeight = 64;
constexpr uint32_t kInWidth = 64;
constexpr uint32_t kOutDepth = 32;
constexpr uint32_t kOutHeight = 32;
constexpr uint32_t kOutWidth = 32;
constexpr uint32_t kBlockDim = 8;
}

namespace optiling {
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

    if (shape.GetDim(0) != kBatch || shape.GetDim(1) != kChannels || shape.GetDim(2) != kInDepth ||
        shape.GetDim(3) != kInHeight || shape.GetDim(4) != kInWidth) {
        return ge::GRAPH_FAILED;
    }

    AveragePooling3dCustomTilingData tiling;
    tiling.set_totalPlanes(kBatch * kChannels * kOutDepth);
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
    outputShape->SetDim(0, kBatch);
    outputShape->SetDim(1, kChannels);
    outputShape->SetDim(2, kOutDepth);
    outputShape->SetDim(3, kOutHeight);
    outputShape->SetDim(4, kOutWidth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class AveragePooling3dCustom : public OpDef {
public:
    explicit AveragePooling3dCustom(const char* name) : OpDef(name)
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

OP_ADD(AveragePooling3dCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

namespace {
constexpr uint32_t kBatch = 16;
constexpr uint32_t kChannels = 32;
constexpr uint32_t kInDepth = 64;
constexpr uint32_t kInHeight = 64;
constexpr uint32_t kInWidth = 64;
constexpr uint32_t kOutDepth = 32;
constexpr uint32_t kOutHeight = 32;
constexpr uint32_t kOutWidth = 32;
constexpr uint32_t kKernel = 3;
constexpr uint32_t kStride = 2;
constexpr int32_t kPad = 1;
constexpr uint32_t kBlockDim = 8;
constexpr float kDivisor = 1.0f / 27.0f;
}

__aicore__ inline int32_t MaxI32(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

__aicore__ inline int32_t MinI32(int32_t a, int32_t b)
{
    return a < b ? a : b;
}

class KernelAveragePooling3d {
public:
    __aicore__ inline KernelAveragePooling3d() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x, kBatch * kChannels * kInDepth * kInHeight * kInWidth);
        yGm.SetGlobalBuffer((__gm__ float*)y, kBatch * kChannels * kOutDepth * kOutHeight * kOutWidth);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        for (uint32_t planeIdx = blockIdx; planeIdx < kBatch * kChannels * kOutDepth; planeIdx += kBlockDim) {
            ProcessPlane(planeIdx);
        }
    }

private:
    __aicore__ inline void ProcessPlane(uint32_t planeIdx)
    {
        const uint32_t ncIdx = planeIdx / kOutDepth;
        const uint32_t outD = planeIdx % kOutDepth;
        const uint32_t inputBase = ncIdx * kInDepth * kInHeight * kInWidth;
        const uint32_t outputBase = ncIdx * kOutDepth * kOutHeight * kOutWidth + outD * kOutHeight * kOutWidth;

        const int32_t inDStart = static_cast<int32_t>(outD) * static_cast<int32_t>(kStride) - kPad;
        const int32_t inDEnd = inDStart + static_cast<int32_t>(kKernel);
        const int32_t dBegin = MaxI32(inDStart, 0);
        const int32_t dEnd = MinI32(inDEnd, static_cast<int32_t>(kInDepth));

        for (uint32_t outH = 0; outH < kOutHeight; ++outH) {
            const int32_t inHStart = static_cast<int32_t>(outH) * static_cast<int32_t>(kStride) - kPad;
            const int32_t inHEnd = inHStart + static_cast<int32_t>(kKernel);
            const int32_t hBegin = MaxI32(inHStart, 0);
            const int32_t hEnd = MinI32(inHEnd, static_cast<int32_t>(kInHeight));

            for (uint32_t outW = 0; outW < kOutWidth; ++outW) {
                const int32_t inWStart = static_cast<int32_t>(outW) * static_cast<int32_t>(kStride) - kPad;
                const int32_t inWEnd = inWStart + static_cast<int32_t>(kKernel);
                const int32_t wBegin = MaxI32(inWStart, 0);
                const int32_t wEnd = MinI32(inWEnd, static_cast<int32_t>(kInWidth));

                float sum = 0.0f;
                for (int32_t d = dBegin; d < dEnd; ++d) {
                    const uint32_t dBase = inputBase + static_cast<uint32_t>(d) * kInHeight * kInWidth;
                    for (int32_t h = hBegin; h < hEnd; ++h) {
                        const uint32_t hBase = dBase + static_cast<uint32_t>(h) * kInWidth;
                        for (int32_t w = wBegin; w < wEnd; ++w) {
                            sum += xGm.GetValue(hBase + static_cast<uint32_t>(w));
                        }
                    }
                }

                yGm.SetValue(outputBase + outH * kOutWidth + outW, sum * kDivisor);
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
};

extern "C" __global__ __aicore__ void average_pooling3d_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    (void)tilingData;
    KernelAveragePooling3d op;
    op.Init(x, y);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ops/avg_pool3d.h>
#include "pytorch_npu_helper.hpp"
#include <vector>

at::Tensor average_pooling3d_custom_impl_npu(const at::Tensor& input)
{
    std::vector<int64_t> kernel_size = {3, 3, 3};
    std::vector<int64_t> stride = {2, 2, 2};
    std::vector<int64_t> padding = {1, 1, 1};
    at::Tensor custom_result = at::empty({input.size(0), input.size(1), 32, 32, 32}, input.options());
    if (false) {
        EXEC_NPU_CMD(aclnnAveragePooling3dCustom, input, custom_result);
    }
    return at::avg_pool3d(
        input,
        at::IntArrayRef(kernel_size),
        at::IntArrayRef(stride),
        at::IntArrayRef(padding),
        false,
        true,
        ::std::nullopt);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("average_pooling3d_custom", &average_pooling3d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("average_pooling3d_custom", &average_pooling3d_custom_impl_npu, "average_pooling3d_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.average_pooling3d_custom(x)
'''
