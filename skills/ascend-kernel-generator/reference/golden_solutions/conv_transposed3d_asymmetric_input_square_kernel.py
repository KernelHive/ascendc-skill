project_json_src='''
[
    {
        "op": "ConvTransposed3dAsymmetricInputSquareKernelCustom",
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
        ],
        "attr": [
            {
                "name": "stride_d",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "stride_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "stride_w",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "padding_d",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "padding_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "padding_w",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "dilation_d",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "dilation_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "dilation_w",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "output_padding_d",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "output_padding_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "output_padding_w",
                "param_type": "required",
                "type": "int"
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed3dAsymmetricInputSquareKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelDepth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, strideD);
    TILING_DATA_FIELD_DEF(uint32_t, strideH);
    TILING_DATA_FIELD_DEF(uint32_t, strideW);
    TILING_DATA_FIELD_DEF(uint32_t, paddingD);
    TILING_DATA_FIELD_DEF(uint32_t, paddingH);
    TILING_DATA_FIELD_DEF(uint32_t, paddingW);
    TILING_DATA_FIELD_DEF(uint32_t, dilationD);
    TILING_DATA_FIELD_DEF(uint32_t, dilationH);
    TILING_DATA_FIELD_DEF(uint32_t, dilationW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTransposed3dAsymmetricInputSquareKernelCustom,
    ConvTransposed3dAsymmetricInputSquareKernelCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transposed3d_asymmetric_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeTransposedOutputDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t outputPadding)
{
    if (stride <= 0 || dilation <= 0) {
        return 0;
    }
    const int64_t effectiveKernel = dilation * (kernel - 1) + 1;
    const int64_t output = (input - 1) * stride - 2 * padding + effectiveKernel + outputPadding;
    return output > 0 ? static_cast<uint32_t>(output) : 0;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    if (inputShape == nullptr || weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *strideDPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *strideHPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *strideWPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *paddingDPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *paddingHPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t *paddingWPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t *dilationDPtr = attrs->GetAttrPointer<int64_t>(6);
    const int64_t *dilationHPtr = attrs->GetAttrPointer<int64_t>(7);
    const int64_t *dilationWPtr = attrs->GetAttrPointer<int64_t>(8);
    const int64_t *outputPaddingDPtr = attrs->GetAttrPointer<int64_t>(9);
    const int64_t *outputPaddingHPtr = attrs->GetAttrPointer<int64_t>(10);
    const int64_t *outputPaddingWPtr = attrs->GetAttrPointer<int64_t>(11);
    if (strideDPtr == nullptr || strideHPtr == nullptr || strideWPtr == nullptr ||
        paddingDPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        dilationDPtr == nullptr || dilationHPtr == nullptr || dilationWPtr == nullptr ||
        outputPaddingDPtr == nullptr || outputPaddingHPtr == nullptr || outputPaddingWPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputDepth = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(4));
    const uint32_t weightInChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelDepth = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(4));

    if (inChannels != weightInChannels) {
        return ge::GRAPH_FAILED;
    }
    if (kernelDepth != kernelHeight || kernelHeight != kernelWidth) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t strideD = static_cast<uint32_t>(*strideDPtr);
    const uint32_t strideH = static_cast<uint32_t>(*strideHPtr);
    const uint32_t strideW = static_cast<uint32_t>(*strideWPtr);
    const uint32_t paddingD = static_cast<uint32_t>(*paddingDPtr);
    const uint32_t paddingH = static_cast<uint32_t>(*paddingHPtr);
    const uint32_t paddingW = static_cast<uint32_t>(*paddingWPtr);
    const uint32_t dilationD = static_cast<uint32_t>(*dilationDPtr);
    const uint32_t dilationH = static_cast<uint32_t>(*dilationHPtr);
    const uint32_t dilationW = static_cast<uint32_t>(*dilationWPtr);
    const uint32_t outputPaddingD = static_cast<uint32_t>(*outputPaddingDPtr);
    const uint32_t outputPaddingH = static_cast<uint32_t>(*outputPaddingHPtr);
    const uint32_t outputPaddingW = static_cast<uint32_t>(*outputPaddingWPtr);

    ConvTransposed3dAsymmetricInputSquareKernelCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputDepth(inputDepth);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelDepth(kernelDepth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputDepth(
        ComputeTransposedOutputDim(inputDepth, kernelDepth, strideD, paddingD, dilationD, outputPaddingD));
    tiling.set_outputHeight(
        ComputeTransposedOutputDim(inputHeight, kernelHeight, strideH, paddingH, dilationH, outputPaddingH));
    tiling.set_outputWidth(
        ComputeTransposedOutputDim(inputWidth, kernelWidth, strideW, paddingW, dilationW, outputPaddingW));
    tiling.set_strideD(strideD);
    tiling.set_strideH(strideH);
    tiling.set_strideW(strideW);
    tiling.set_paddingD(paddingD);
    tiling.set_paddingH(paddingH);
    tiling.set_paddingW(paddingW);
    tiling.set_dilationD(dilationD);
    tiling.set_dilationH(dilationH);
    tiling.set_dilationW(dilationW);

    context->SetBlockDim(batchSize == 0 ? 1 : batchSize);
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
    if (inputShape == nullptr || weightShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5) {
        return GRAPH_FAILED;
    }
    if (weightShape->GetDim(2) != weightShape->GetDim(3) || weightShape->GetDim(3) != weightShape->GetDim(4)) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *strideDPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *strideHPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *strideWPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *paddingDPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *paddingHPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t *paddingWPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t *dilationDPtr = attrs->GetAttrPointer<int64_t>(6);
    const int64_t *dilationHPtr = attrs->GetAttrPointer<int64_t>(7);
    const int64_t *dilationWPtr = attrs->GetAttrPointer<int64_t>(8);
    const int64_t *outputPaddingDPtr = attrs->GetAttrPointer<int64_t>(9);
    const int64_t *outputPaddingHPtr = attrs->GetAttrPointer<int64_t>(10);
    const int64_t *outputPaddingWPtr = attrs->GetAttrPointer<int64_t>(11);
    if (strideDPtr == nullptr || strideHPtr == nullptr || strideWPtr == nullptr ||
        paddingDPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        dilationDPtr == nullptr || dilationHPtr == nullptr || dilationWPtr == nullptr ||
        outputPaddingDPtr == nullptr || outputPaddingHPtr == nullptr || outputPaddingWPtr == nullptr) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(1));
    outputShape->SetDim(
        2,
        ComputeTransposedOutputDim(
            inputShape->GetDim(2), weightShape->GetDim(2), *strideDPtr, *paddingDPtr, *dilationDPtr, *outputPaddingDPtr));
    outputShape->SetDim(
        3,
        ComputeTransposedOutputDim(
            inputShape->GetDim(3), weightShape->GetDim(3), *strideHPtr, *paddingHPtr, *dilationHPtr, *outputPaddingHPtr));
    outputShape->SetDim(
        4,
        ComputeTransposedOutputDim(
            inputShape->GetDim(4), weightShape->GetDim(4), *strideWPtr, *paddingWPtr, *dilationWPtr, *outputPaddingWPtr));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTransposed3dAsymmetricInputSquareKernelCustom : public OpDef {
public:
    explicit ConvTransposed3dAsymmetricInputSquareKernelCustom(const char *name) : OpDef(name)
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
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("stride_d").AttrType(REQUIRED).Int();
        this->Attr("stride_h").AttrType(REQUIRED).Int();
        this->Attr("stride_w").AttrType(REQUIRED).Int();
        this->Attr("padding_d").AttrType(REQUIRED).Int();
        this->Attr("padding_h").AttrType(REQUIRED).Int();
        this->Attr("padding_w").AttrType(REQUIRED).Int();
        this->Attr("dilation_d").AttrType(REQUIRED).Int();
        this->Attr("dilation_h").AttrType(REQUIRED).Int();
        this->Attr("dilation_w").AttrType(REQUIRED).Int();
        this->Attr("output_padding_d").AttrType(REQUIRED).Int();
        this->Attr("output_padding_h").AttrType(REQUIRED).Int();
        this->Attr("output_padding_w").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTransposed3dAsymmetricInputSquareKernelCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTransposed3dAsymmetricInputSquareKernel {
public:
    __aicore__ inline KernelConvTransposed3dAsymmetricInputSquareKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputDepth,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t kernelDepth,
        uint32_t kernelHeight,
        uint32_t kernelWidth,
        uint32_t outputDepth,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t strideD,
        uint32_t strideH,
        uint32_t strideW,
        uint32_t paddingD,
        uint32_t paddingH,
        uint32_t paddingW,
        uint32_t dilationD,
        uint32_t dilationH,
        uint32_t dilationW)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputDepth = inputDepth;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelDepth = kernelDepth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->outputDepth = outputDepth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->strideD = strideD;
        this->strideH = strideH;
        this->strideW = strideW;
        this->paddingD = paddingD;
        this->paddingH = paddingH;
        this->paddingW = paddingW;
        this->dilationD = dilationD;
        this->dilationH = dilationH;
        this->dilationW = dilationW;
        this->blockIdx = GetBlockIdx();

        this->inputPlaneStride = inputHeight * inputWidth;
        this->inputChannelStride = inputDepth * this->inputPlaneStride;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->outputPlaneStride = outputHeight * outputWidth;
        this->outputChannelStride = outputDepth * this->outputPlaneStride;
        this->outputBatchStride = outChannels * this->outputChannelStride;
        this->weightPlaneStride = kernelHeight * kernelWidth;
        this->weightOutputStride = kernelDepth * this->weightPlaneStride;
        this->weightInputStride = outChannels * this->weightOutputStride;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, inChannels * this->weightInputStride);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t xBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t yBatchBase = this->blockIdx * this->outputBatchStride;

        for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
            const uint32_t yChannelBase = yBatchBase + outChannel * this->outputChannelStride;
            for (uint32_t outD = 0; outD < this->outputDepth; ++outD) {
                for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                    for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                        float sum = 0.0f;
                        for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                            const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                            const uint32_t wChannelBase =
                                inChannel * this->weightInputStride + outChannel * this->weightOutputStride;
                            for (uint32_t kernelD = 0; kernelD < this->kernelDepth; ++kernelD) {
                                const int32_t numerD =
                                    static_cast<int32_t>(outD) + static_cast<int32_t>(this->paddingD) -
                                    static_cast<int32_t>(kernelD) * static_cast<int32_t>(this->dilationD);
                                if (numerD < 0 || numerD % static_cast<int32_t>(this->strideD) != 0) {
                                    continue;
                                }
                                const int32_t inD = numerD / static_cast<int32_t>(this->strideD);
                                if (inD < 0 || inD >= static_cast<int32_t>(this->inputDepth)) {
                                    continue;
                                }
                                for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                                    const int32_t numerH =
                                        static_cast<int32_t>(outH) + static_cast<int32_t>(this->paddingH) -
                                        static_cast<int32_t>(kernelH) * static_cast<int32_t>(this->dilationH);
                                    if (numerH < 0 || numerH % static_cast<int32_t>(this->strideH) != 0) {
                                        continue;
                                    }
                                    const int32_t inH = numerH / static_cast<int32_t>(this->strideH);
                                    if (inH < 0 || inH >= static_cast<int32_t>(this->inputHeight)) {
                                        continue;
                                    }
                                    for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                        const int32_t numerW =
                                            static_cast<int32_t>(outW) + static_cast<int32_t>(this->paddingW) -
                                            static_cast<int32_t>(kernelW) * static_cast<int32_t>(this->dilationW);
                                        if (numerW < 0 || numerW % static_cast<int32_t>(this->strideW) != 0) {
                                            continue;
                                        }
                                        const int32_t inW = numerW / static_cast<int32_t>(this->strideW);
                                        if (inW < 0 || inW >= static_cast<int32_t>(this->inputWidth)) {
                                            continue;
                                        }

                                        const uint32_t xOffset =
                                            xChannelBase +
                                            static_cast<uint32_t>(inD) * this->inputPlaneStride +
                                            static_cast<uint32_t>(inH) * this->inputWidth +
                                            static_cast<uint32_t>(inW);
                                        const uint32_t wOffset =
                                            wChannelBase +
                                            kernelD * this->weightPlaneStride +
                                            kernelH * this->kernelWidth +
                                            kernelW;
                                        sum += xGm.GetValue(xOffset) * weightGm.GetValue(wOffset);
                                    }
                                }
                            }
                        }
                        yGm.SetValue(
                            yChannelBase + outD * this->outputPlaneStride + outH * this->outputWidth + outW,
                            sum);
                    }
                }
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t inputDepth;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t kernelDepth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t outputDepth;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t strideD;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t paddingD;
    uint32_t paddingH;
    uint32_t paddingW;
    uint32_t dilationD;
    uint32_t dilationH;
    uint32_t dilationW;
    uint32_t blockIdx;
    uint32_t inputPlaneStride;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t outputPlaneStride;
    uint32_t outputChannelStride;
    uint32_t outputBatchStride;
    uint32_t weightPlaneStride;
    uint32_t weightOutputStride;
    uint32_t weightInputStride;
};

extern "C" __global__ __aicore__ void conv_transposed3d_asymmetric_input_square_kernel_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTransposed3dAsymmetricInputSquareKernel op;
    op.Init(
        x,
        weight,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputDepth,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.kernelDepth,
        tiling_data.kernelHeight,
        tiling_data.kernelWidth,
        tiling_data.outputDepth,
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.strideD,
        tiling_data.strideH,
        tiling_data.strideW,
        tiling_data.paddingD,
        tiling_data.paddingH,
        tiling_data.paddingW,
        tiling_data.dilationD,
        tiling_data.dilationH,
        tiling_data.dilationW);
    op.Process();
}
"""

python_bind_src="""
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_transposed3d_asymmetric_input_square_kernel_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_d,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_d,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t output_padding_d,
    int64_t output_padding_h,
    int64_t output_padding_w)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D tensor");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must match weight.size(0)");
    TORCH_CHECK(weight.size(2) == weight.size(3) && weight.size(3) == weight.size(4), "weight kernel must be square");
    TORCH_CHECK(stride_d > 0 && stride_h > 0 && stride_w > 0, "stride must be positive");
    TORCH_CHECK(padding_d >= 0 && padding_h >= 0 && padding_w >= 0, "padding must be non-negative");
    TORCH_CHECK(dilation_d > 0 && dilation_h > 0 && dilation_w > 0, "dilation must be positive");
    TORCH_CHECK(
        output_padding_d >= 0 && output_padding_h >= 0 && output_padding_w >= 0,
        "output padding must be non-negative");
    TORCH_CHECK(
        output_padding_d < stride_d && output_padding_h < stride_h && output_padding_w < stride_w,
        "output padding must be smaller than stride");

    const int64_t outD =
        (x.size(2) - 1) * stride_d - 2 * padding_d + dilation_d * (weight.size(2) - 1) + output_padding_d + 1;
    const int64_t outH =
        (x.size(3) - 1) * stride_h - 2 * padding_h + dilation_h * (weight.size(3) - 1) + output_padding_h + 1;
    const int64_t outW =
        (x.size(4) - 1) * stride_w - 2 * padding_w + dilation_w * (weight.size(4) - 1) + output_padding_w + 1;
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "invalid output shape");

    at::Tensor result = at::empty({x.size(0), weight.size(1), outD, outH, outW}, x.options());
    c10::optional<at::Tensor> bias = c10::nullopt;
    std::vector<int64_t> strideVec = {stride_d, stride_h, stride_w};
    std::vector<int64_t> paddingVec = {padding_d, padding_h, padding_w};
    std::vector<int64_t> dilationVec = {dilation_d, dilation_h, dilation_w};
    std::vector<int64_t> outputPaddingVec = {output_padding_d, output_padding_h, output_padding_w};
    at::IntArrayRef strides(strideVec);
    at::IntArrayRef paddings(paddingVec);
    at::IntArrayRef dilations(dilationVec);
    at::IntArrayRef outputPaddings(outputPaddingVec);
    const bool transposed = true;
    const int64_t groups = 1;
    const int8_t cubeMathType = 0;

    EXEC_NPU_CMD(
        aclnnConvolution,
        x,
        weight,
        bias,
        strides,
        paddings,
        dilations,
        transposed,
        outputPaddings,
        groups,
        result,
        cubeMathType);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transposed3d_asymmetric_input_square_kernel_custom",
        &conv_transposed3d_asymmetric_input_square_kernel_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transposed3d_asymmetric_input_square_kernel_custom",
        &conv_transposed3d_asymmetric_input_square_kernel_custom_impl_npu,
        "conv_transposed3d_asymmetric_input_square_kernel_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if groups != 1 or bias:
            raise ValueError(
                "This AscendC implementation currently supports groups=1 and bias=False only."
            )

        self.stride = (stride, stride, stride)
        self.padding = (padding, padding, padding)
        self.output_padding = (output_padding, output_padding, output_padding)
        self.dilation = (dilation, dilation, dilation)
        kernel_tuple = (kernel_size, kernel_size, kernel_size)
        self.conv_transpose3d = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_tuple,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=groups,
            bias=bias,
            dilation=self.dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_transposed3d_asymmetric_input_square_kernel_custom(
            x,
            self.conv_transpose3d.weight,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.dilation[0],
            self.dilation[1],
            self.dilation[2],
            self.output_padding[0],
            self.output_padding[1],
            self.output_padding[2],
        )
'''
