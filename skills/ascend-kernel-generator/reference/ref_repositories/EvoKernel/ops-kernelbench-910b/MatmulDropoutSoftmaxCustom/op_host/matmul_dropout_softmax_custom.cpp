
#include "matmul_dropout_softmax_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static constexpr uint32_t kMaxBlockDim = 24;

// choose tileN aligned for burst; keep UB safe
static constexpr uint32_t kAlignN = 256;
static constexpr uint32_t kDefaultTileN = 1024;
static constexpr uint32_t kMinTileN = 256;
static constexpr uint32_t kMaxTileN = 2048;

// chunk K for UB; keep moderate
static constexpr uint32_t kDefaultChunkK = 256;
static constexpr uint32_t kMaxChunkK = 512;

static inline uint32_t AlignDown(uint32_t v, uint32_t a) {
  return (a == 0) ? v : (v / a) * a;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulDropoutSoftmaxCustomTilingData tiling;

    const auto* xShapeS = context->GetInputShape(0);
    const auto* wShapeS = context->GetInputShape(1);
    const auto* bShapeS = context->GetInputShape(2);
    if (xShapeS == nullptr || wShapeS == nullptr || bShapeS == nullptr) return ge::GRAPH_FAILED;

    const gert::Shape& xShape = xShapeS->GetOriginShape();
    const gert::Shape& wShape = wShapeS->GetOriginShape();
    const gert::Shape& bShape = bShapeS->GetOriginShape();

    // x:[B,K], w:[N,K], b:[N]
    if (xShape.GetDimNum() != 2 || wShape.GetDimNum() != 2 || bShape.GetDimNum() != 1) return ge::GRAPH_FAILED;

    const uint64_t B64 = static_cast<uint64_t>(xShape.GetDim(0));
    const uint64_t K64 = static_cast<uint64_t>(xShape.GetDim(1));
    const uint64_t N64 = static_cast<uint64_t>(wShape.GetDim(0));
    const uint64_t K2  = static_cast<uint64_t>(wShape.GetDim(1));
    const uint64_t BN  = static_cast<uint64_t>(bShape.GetDim(0));
    if (K2 != K64) return ge::GRAPH_FAILED;
    if (BN != N64) return ge::GRAPH_FAILED;

    const uint32_t B = static_cast<uint32_t>(B64);
    const uint32_t K = static_cast<uint32_t>(K64);
    const uint32_t N = static_cast<uint32_t>(N64);

    uint32_t blockDim = std::min<uint32_t>(kMaxBlockDim, std::max<uint32_t>(1u, (B == 0) ? 1u : B));
    context->SetBlockDim(blockDim);

    uint32_t rowsPerCore = (B + blockDim - 1u) / blockDim;
    if (rowsPerCore == 0) rowsPerCore = 1;

    uint32_t tileN = kDefaultTileN;
    if (N > 0 && tileN > N) tileN = N;
    tileN = std::min<uint32_t>(tileN, kMaxTileN);
    tileN = std::max<uint32_t>(tileN, kMinTileN);
    tileN = AlignDown(tileN, kAlignN);
    if (tileN == 0) tileN = (N > 0) ? std::min<uint32_t>(N, kAlignN) : 1;
    if (N > 0 && tileN > N) tileN = N;
    if (N == 0) tileN = 1;

    uint32_t kChunkK = kDefaultChunkK;
    if (K < kChunkK) kChunkK = K;
    if (kChunkK == 0) kChunkK = 1;
    if (kChunkK > kMaxChunkK) kChunkK = kMaxChunkK;

    tiling.set_B(B);
    tiling.set_K(K);
    tiling.set_N(N);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_tileN(tileN);
    tiling.set_kChunkK(kChunkK);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // workspace:
    // ws0: logits workspace [B*N] float
    // ws1: rowMax [B] float
    // ws2: rowSum [B] float
    size_t* ws = context->GetWorkspaceSizes(3);
    ws[0] = static_cast<size_t>(B64) * static_cast<size_t>(N64) * sizeof(float);
    ws[1] = static_cast<size_t>(B64) * sizeof(float);
    ws[2] = static_cast<size_t>(B64) * sizeof(float);
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    const gert::Shape* w_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x_shape == nullptr || w_shape == nullptr || y_shape == nullptr) return GRAPH_FAILED;
    if (x_shape->GetDimNum() != 2 || w_shape->GetDimNum() != 2) return GRAPH_FAILED;
    const int64_t B = x_shape->GetDim(0);
    const int64_t N = w_shape->GetDim(0);
    y_shape->SetDimNum(2);
    y_shape->SetDim(0, B);
    y_shape->SetDim(1, N);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) return GRAPH_FAILED;
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class MatmulDropoutSoftmaxCustom : public OpDef {
public:
    explicit MatmulDropoutSoftmaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("dropout_p").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("training").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MatmulDropoutSoftmaxCustom);

}  // namespace ops
