
#include "conv3d_leaky_relu_sum_clamp_gelu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static inline int64_t out_dim_floor(int64_t in, int64_t k, int64_t pad, int64_t stride, int64_t dil) {
    const int64_t eff = dil * (k - 1) + 1;
    if (in + 2 * pad < eff) return -1;
    return (in + 2 * pad - eff) / stride + 1;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dLeakyReluSumClampGeluCustomTilingData tiling;

    const auto xShape = context->GetInputShape(0)->GetStorageShape();
    const auto wShape = context->GetInputShape(1)->GetStorageShape();
    const auto bShape = context->GetInputShape(2)->GetStorageShape();
    const auto sShape = context->GetInputShape(3)->GetStorageShape();

    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5 || bShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t n   = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t cin = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t din = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t hin = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t win = static_cast<uint32_t>(xShape.GetDim(4));

    const uint32_t cout = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t wcin = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kd   = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kh   = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t kw   = static_cast<uint32_t>(wShape.GetDim(4));

    if (static_cast<uint32_t>(bShape.GetDim(0)) != cout) return ge::GRAPH_FAILED;

    constexpr uint32_t STR = 1;
    constexpr uint32_t PAD = 0;
    constexpr uint32_t DIL = 1;
    constexpr uint32_t GRP = 1;

    if (GRP != 1) return ge::GRAPH_FAILED;
    if (wcin != cin) return ge::GRAPH_FAILED;
    if (kd != 3 || kh != 3 || kw != 3) return ge::GRAPH_FAILED;

    const int64_t dout64 = out_dim_floor(static_cast<int64_t>(din), kd, PAD, STR, DIL);
    const int64_t hout64 = out_dim_floor(static_cast<int64_t>(hin), kh, PAD, STR, DIL);
    const int64_t wout64 = out_dim_floor(static_cast<int64_t>(win), kw, PAD, STR, DIL);
    if (dout64 <= 0 || hout64 <= 0 || wout64 <= 0) return ge::GRAPH_FAILED;

    const uint32_t dout = static_cast<uint32_t>(dout64);
    const uint32_t hout = static_cast<uint32_t>(hout64);
    const uint32_t wout = static_cast<uint32_t>(wout64);

    if (static_cast<uint64_t>(sShape.GetShapeSize()) != static_cast<uint64_t>(cout)) return ge::GRAPH_FAILED;

    if (!(n == 128 && cin == 8 && cout == 64 && din == 16 && hin == 64 && win == 64)) {
        return ge::GRAPH_FAILED;
    }
    if (!(dout == 14 && hout == 62 && wout == 62)) return ge::GRAPH_FAILED;

    tiling.set_n(n);
    tiling.set_cin(cin);
    tiling.set_din(din);
    tiling.set_hin(hin);
    tiling.set_win(win);

    tiling.set_cout(cout);
    tiling.set_kd(kd);
    tiling.set_kh(kh);
    tiling.set_kw(kw);

    tiling.set_dout(dout);
    tiling.set_hout(hout);
    tiling.set_wout(wout);

    tiling.set_stride(STR);
    tiling.set_pad(PAD);
    tiling.set_dilation(DIL);
    tiling.set_groups(GRP);

    tiling.set_leaky_slope(0.2f);
    tiling.set_clamp_min(-1.0f);
    tiling.set_clamp_max(1.0f);

    const uint64_t total_tasks64 = static_cast<uint64_t>(n) * cout * dout * hout * wout;
    if (total_tasks64 > 0xFFFFFFFFull) return ge::GRAPH_FAILED;
    const uint32_t total_tasks = static_cast<uint32_t>(total_tasks64);
    tiling.set_total_tasks(total_tasks);

    // New stable mapping: fixed work quota per block for better occupancy and shorter block runtimes.
    // Keep it conservative to avoid resource pressure.
    constexpr uint32_t TASKS_PER_BLOCK = 8192;  // ~32 tiles of 256
    tiling.set_tasks_per_block(TASKS_PER_BLOCK);

    const uint32_t grid = (total_tasks + TASKS_PER_BLOCK - 1) / TASKS_PER_BLOCK;
    context->SetBlockDim(grid);

    tiling.set_total_x(static_cast<uint32_t>(xShape.GetShapeSize()));
    tiling.set_total_w(static_cast<uint32_t>(wShape.GetShapeSize()));
    tiling.set_total_b(static_cast<uint32_t>(bShape.GetShapeSize()));
    tiling.set_total_s(static_cast<uint32_t>(sShape.GetShapeSize()));
    tiling.set_total_y(static_cast<uint32_t>(context->GetOutputShape(0)->GetStorageShape().GetShapeSize()));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class Conv3dLeakyReluSumClampGeluCustom : public OpDef {
public:
    explicit Conv3dLeakyReluSumClampGeluCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sum_tensor").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dLeakyReluSumClampGeluCustom);

} // namespace ops
