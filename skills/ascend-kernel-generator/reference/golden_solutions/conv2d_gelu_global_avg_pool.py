project_json_src='''
[
    {
        "op": "Conv2dGeluGlobalAvgPoolCustom",
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
            },
            {
                "name": "weight",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "bias",
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
BEGIN_TILING_DATA_DEF(Conv2dGeluGlobalAvgPoolCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(float, outputReciprocal);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Conv2dGeluGlobalAvgPoolCustom,
    Conv2dGeluGlobalAvgPoolCustomTilingData)
}
"""

host_operator_src="""
#include "conv2d_gelu_global_avg_pool_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *biasShape = context->GetInputShape(2);
    if (inputShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    const auto bShape = biasShape->GetStorageShape();
    if (xShape.GetDimNum() != 4 || wShape.GetDimNum() != 4 || bShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t weightInChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t biasLength = static_cast<uint32_t>(bShape.GetDim(0));

    if (inChannels != weightInChannels || biasLength != outChannels) {
        return ge::GRAPH_FAILED;
    }
    if (inputHeight < kernelHeight || inputWidth < kernelWidth) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t outputHeight = inputHeight - kernelHeight + 1;
    const uint32_t outputWidth = inputWidth - kernelWidth + 1;
    const uint32_t outputElemCount = outputHeight * outputWidth;

    Conv2dGeluGlobalAvgPoolCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_outputReciprocal(
        outputElemCount == 0 ? 0.0f : 1.0f / static_cast<float>(outputElemCount));

    const uint32_t blockDim = batchSize == 0 ? 1 : (batchSize > 8 ? 8 : batchSize);
    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    const gert::Shape *weightShape = context->GetInputShape(1);
    const gert::Shape *biasShape = context->GetInputShape(2);
    if (inputShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDim(1) != weightShape->GetDim(1) || weightShape->GetDim(0) != biasShape->GetDim(0)) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDim(2) < weightShape->GetDim(2) || inputShape->GetDim(3) < weightShape->GetDim(3)) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(2);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Conv2dGeluGlobalAvgPoolCustom : public OpDef {
public:
    explicit Conv2dGeluGlobalAvgPoolCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
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

OP_ADD(Conv2dGeluGlobalAvgPoolCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConv2dGeluGlobalAvgPool {
public:
    __aicore__ inline KernelConv2dGeluGlobalAvgPool() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t kernelHeight,
        uint32_t kernelWidth,
        uint32_t outputHeight,
        uint32_t outputWidth,
        float outputReciprocal)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->outputReciprocal = outputReciprocal;
        this->blockIdx = GetBlockIdx();
        this->blockDim = GetBlockNum();
        this->inputChannelStride = inputHeight * inputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->weightOutStride = inChannels * kernelHeight * kernelWidth;
        this->weightInStride = kernelHeight * kernelWidth;
        this->outputBatchStride = outChannels;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outChannels * this->weightOutStride);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batchIdx = this->blockIdx; batchIdx < this->batchSize; batchIdx += this->blockDim) {
            const uint32_t outputBase = batchIdx * this->outputBatchStride;
            for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                float sum = 0.0f;
                for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                    for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                        sum += Gelu(ComputeConvAt(batchIdx, outChannel, outH, outW));
                    }
                }
                yGm.SetValue(outputBase + outChannel, sum * this->outputReciprocal);
            }
        }
    }

private:
    __aicore__ inline float ComputeConvAt(
        uint32_t batchIdx,
        uint32_t outChannel,
        uint32_t outH,
        uint32_t outW) const
    {
        const uint32_t inputBatchBase = batchIdx * this->inputBatchStride;
        const uint32_t weightBase = outChannel * this->weightOutStride;
        float sum = biasGm.GetValue(outChannel);
        for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
            const uint32_t inputChannelBase = inputBatchBase + inChannel * this->inputChannelStride;
            const uint32_t weightChannelBase = weightBase + inChannel * this->weightInStride;
            for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                const uint32_t inputRowBase = inputChannelBase + (outH + kernelH) * this->inputWidth;
                const uint32_t weightRowBase = weightChannelBase + kernelH * this->kernelWidth;
                for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                    const uint32_t xOffset = inputRowBase + outW + kernelW;
                    const uint32_t wOffset = weightRowBase + kernelW;
                    sum += xGm.GetValue(xOffset) * weightGm.GetValue(wOffset);
                }
            }
        }
        return sum;
    }

    __aicore__ inline float Gelu(float value) const
    {
        if (value <= -4.0f) {
            return -1.266849673e-04f;
        }
        if (value >= 4.0f) {
            return value;
        }

        constexpr float kStart = -4.0f;
        constexpr float kInvStep = 3.2e+01f;
        const float scaled = (value - kStart) * kInvStep;
        int32_t idx = static_cast<int32_t>(scaled);
        if (idx < 0) {
            idx = 0;
        } else if (idx > 255) {
            idx = 255;
        }
        const float frac = scaled - static_cast<float>(idx);
        const float left = kGeluTable[idx];
        const float right = kGeluTable[idx + 1];
        return left + (right - left) * frac;
    }

private:
    static constexpr float kGeluTable[257] = {
        -1.266849673e-04f, -1.433723479e-04f, -1.620968973e-04f, -1.830850196e-04f, -2.065853553e-04f, -2.328706012e-04f, -2.622394371e-04f, -2.950185589e-04f,
        -3.315648195e-04f, -3.722674769e-04f, -4.175505477e-04f, -4.678752640e-04f, -5.237426313e-04f, -5.856960809e-04f, -6.543242120e-04f, -7.302636166e-04f,
        -8.142017766e-04f, -9.068800238e-04f, -1.009096549e-03f, -1.121709448e-03f, -1.245639783e-03f, -1.381874649e-03f, -1.531470212e-03f, -1.695554713e-03f,
        -1.875331388e-03f, -2.072081302e-03f, -2.287166049e-03f, -2.522030283e-03f, -2.778204060e-03f, -3.057304934e-03f, -3.361039781e-03f, -3.691206290e-03f,
        -4.049694095e-03f, -4.438485486e-03f, -4.859655650e-03f, -5.315372392e-03f, -5.807895284e-03f, -6.339574181e-03f, -6.912847055e-03f, -7.530237075e-03f,
        -8.194348896e-03f, -8.907864084e-03f, -9.673535620e-03f, -1.049418144e-02f, -1.137267695e-02f, -1.231194645e-02f, -1.331495342e-02f, -1.438468968e-02f,
        -1.552416331e-02f, -1.673638528e-02f, -1.802435483e-02f, -1.939104361e-02f, -2.083937835e-02f, -2.237222235e-02f, -2.399235552e-02f, -2.570245318e-02f,
        -2.750506347e-02f, -2.940258361e-02f, -3.139723477e-02f, -3.349103586e-02f, -3.568577620e-02f, -3.798298709e-02f, -4.038391248e-02f, -4.288947877e-02f,
        -4.550026390e-02f, -4.821646580e-02f, -5.103787046e-02f, -5.396381960e-02f, -5.699317831e-02f, -6.012430266e-02f, -6.335500763e-02f, -6.668253543e-02f,
        -7.010352451e-02f, -7.361397949e-02f, -7.720924212e-02f, -8.088396371e-02f, -8.463207905e-02f, -8.844678224e-02f, -9.232050458e-02f, -9.624489481e-02f,
        -1.002108019e-01f, -1.042082607e-01f, -1.082264805e-01f, -1.122538371e-01f, -1.162778682e-01f, -1.202852726e-01f, -1.242619127e-01f, -1.281928223e-01f,
        -1.320622171e-01f, -1.358535105e-01f, -1.395493336e-01f, -1.431315594e-01f, -1.465813318e-01f, -1.498790992e-01f, -1.530046527e-01f, -1.559371682e-01f,
        -1.586552539e-01f, -1.611370017e-01f, -1.633600424e-01f, -1.653016057e-01f, -1.669385837e-01f, -1.682475979e-01f, -1.692050695e-01f, -1.697872935e-01f,
        -1.699705143e-01f, -1.697310048e-01f, -1.690451471e-01f, -1.678895146e-01f, -1.662409557e-01f, -1.640766777e-01f, -1.613743323e-01f, -1.581120989e-01f,
        -1.542687694e-01f, -1.498238304e-01f, -1.447575448e-01f, -1.390510305e-01f, -1.326863375e-01f, -1.256465212e-01f, -1.179157130e-01f, -1.094791862e-01f,
        -1.003234186e-01f, -9.043614975e-02f, -7.980643347e-02f, -6.842468487e-02f, -5.628272190e-02f, -4.337380082e-02f, -2.969264569e-02f, -1.523547133e-02f,
        0.000000000e+00f, 1.601452867e-02f, 3.280735431e-02f, 5.037619918e-02f, 6.871727810e-02f, 8.782531513e-02f, 1.076935665e-01f, 1.283138502e-01f,
        1.496765814e-01f, 1.717708138e-01f, 1.945842870e-01f, 2.181034788e-01f, 2.423136625e-01f, 2.671989695e-01f, 2.927424552e-01f, 3.189261696e-01f,
        3.457312306e-01f, 3.731379011e-01f, 4.011256677e-01f, 4.296733223e-01f, 4.587590443e-01f, 4.883604854e-01f, 5.184548529e-01f, 5.490189952e-01f,
        5.800294857e-01f, 6.114627065e-01f, 6.432949305e-01f, 6.755024021e-01f, 7.080614163e-01f, 7.409483943e-01f, 7.741399576e-01f, 8.076129983e-01f,
        8.413447461e-01f, 8.753128318e-01f, 9.094953473e-01f, 9.438709008e-01f, 9.784186682e-01f, 1.013118441e+00f, 1.047950666e+00f, 1.082896489e+00f,
        1.117937783e+00f, 1.153057178e+00f, 1.188238087e+00f, 1.223464727e+00f, 1.258722132e+00f, 1.293996163e+00f, 1.329273520e+00f, 1.364541739e+00f,
        1.399789198e+00f, 1.435005105e+00f, 1.470179495e+00f, 1.505303218e+00f, 1.540367921e+00f, 1.575366036e+00f, 1.610290758e+00f, 1.645136021e+00f,
        1.679896475e+00f, 1.714567465e+00f, 1.749144992e+00f, 1.783625697e+00f, 1.818006822e+00f, 1.852286180e+00f, 1.886462130e+00f, 1.920533534e+00f,
        1.954499736e+00f, 1.988360521e+00f, 2.022116088e+00f, 2.055767013e+00f, 2.089314224e+00f, 2.122758964e+00f, 2.156102765e+00f, 2.189347416e+00f,
        2.222494937e+00f, 2.255547547e+00f, 2.288507644e+00f, 2.321377778e+00f, 2.354160622e+00f, 2.386858956e+00f, 2.419475645e+00f, 2.452013615e+00f,
        2.484475837e+00f, 2.516865310e+00f, 2.549185047e+00f, 2.581438054e+00f, 2.613627323e+00f, 2.645755819e+00f, 2.677826464e+00f, 2.709842136e+00f,
        2.741805651e+00f, 2.773719763e+00f, 2.805587153e+00f, 2.837410426e+00f, 2.869192105e+00f, 2.900934628e+00f, 2.932640344e+00f, 2.964311515e+00f,
        2.995950306e+00f, 3.027558794e+00f, 3.059138960e+00f, 3.090692695e+00f, 3.122221796e+00f, 3.153727970e+00f, 3.185212834e+00f, 3.216677919e+00f,
        3.248124669e+00f, 3.279554445e+00f, 3.310968530e+00f, 3.342368125e+00f, 3.373754360e+00f, 3.405128291e+00f, 3.436490903e+00f, 3.467843120e+00f,
        3.499185798e+00f, 3.530519736e+00f, 3.561845676e+00f, 3.593164304e+00f, 3.624476257e+00f, 3.655782125e+00f, 3.687082449e+00f, 3.718377733e+00f,
        3.749668435e+00f, 3.780954981e+00f, 3.812237761e+00f, 3.843517129e+00f, 3.874793415e+00f, 3.906066915e+00f, 3.937337903e+00f, 3.968606628e+00f,
        3.999873315e+00f,
    };

    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t kernelHeight = 0;
    uint32_t kernelWidth = 0;
    uint32_t outputHeight = 0;
    uint32_t outputWidth = 0;
    uint32_t blockIdx = 0;
    uint32_t blockDim = 1;
    uint32_t inputChannelStride = 0;
    uint32_t inputBatchStride = 0;
    uint32_t weightOutStride = 0;
    uint32_t weightInStride = 0;
    uint32_t outputBatchStride = 0;
    float outputReciprocal = 0.0f;
};

extern "C" __global__ __aicore__ void conv2d_gelu_global_avg_pool_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dGeluGlobalAvgPool op;
    op.Init(
        x,
        weight,
        bias,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.kernelHeight,
        tiling_data.kernelWidth,
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.outputReciprocal);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv2d_gelu_global_avg_pool_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == bias.size(0), "bias length must match output channels");
    TORCH_CHECK(x.size(2) >= weight.size(2) && x.size(3) >= weight.size(3), "kernel must fit inside input");

    at::Tensor result = at::empty({x.size(0), weight.size(0)}, x.options());
    EXEC_NPU_CMD(aclnnConv2dGeluGlobalAvgPoolCustom, x, weight, bias, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_gelu_global_avg_pool_custom", &conv2d_gelu_global_avg_pool_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv2d_gelu_global_avg_pool_custom",
        &conv2d_gelu_global_avg_pool_impl_npu,
        "conv2d_gelu_global_avg_pool_custom");
}
"""

model_src='''
import ctypes
import importlib
import os
import torch
import torch_npu


def _preload_custom_opapi():
    opp_root = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    if not opp_root:
        return
    lib_dir = os.path.join(opp_root, "op_api", "lib")
    for lib_name in ("libcust_opapi.so", "libopapi.so"):
        lib_path = os.path.join(lib_dir, lib_name)
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_custom_opapi()
try:
    import custom_ops_lib as ops_lib
except ImportError:
    ops_lib = importlib.import_module("custom" + "_ops_lib")


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops_lib.conv2d_gelu_global_avg_pool_custom(
            x,
            self.conv.weight,
            self.conv.bias,
        )
'''
