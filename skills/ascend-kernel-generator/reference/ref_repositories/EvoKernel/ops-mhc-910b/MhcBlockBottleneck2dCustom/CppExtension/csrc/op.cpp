
#include <torch/extension.h>
#include <torch/library.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

static inline at::Tensor npu_f16_contig(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().is_privateuseone(), name, " must be on NPU");
    at::Tensor r = t;
    if (r.scalar_type() != at::kHalf) r = r.to(at::kHalf);
    return r.contiguous();
}

static inline at::Tensor npu_i32_scalar(int64_t v, const at::Tensor& like) {
    auto t = at::empty({1}, like.options().dtype(at::kInt));
    t.fill_(static_cast<int32_t>(v));
    return t;
}
static inline at::Tensor npu_f32_scalar(double v, const at::Tensor& like) {
    auto t = at::empty({1}, like.options().dtype(at::kFloat));
    t.fill_(static_cast<float>(v));
    return t;
}

// Fused tail: sinkhorn(logits)->mix(streams)->add(identity)->relu
at::Tensor mhc_block_bottleneck2d_custom_impl_npu(
    const at::Tensor& out_bn3,        // (B,C,H,W) fp16
    const at::Tensor& identity,       // (B,C,H,W) fp16
    const at::Tensor& mapping_logits, // (B,S,S)   fp16 (pre-sinkhorn logits)
    int64_t sinkhorn_iter,
    double sinkhorn_eps,
    double sinkhorn_temperature)
{
    auto out_ = npu_f16_contig(out_bn3, "out_bn3");
    auto id_  = npu_f16_contig(identity, "identity");
    auto map_ = npu_f16_contig(mapping_logits, "mapping_logits");

    TORCH_CHECK(out_.dim() == 4, "out_bn3 must be (B,C,H,W)");
    TORCH_CHECK(id_.sizes() == out_.sizes(), "identity must match out_bn3 shape");
    TORCH_CHECK(map_.dim() == 3, "mapping_logits must be (B,S,S)");
    TORCH_CHECK(map_.size(0) == out_.size(0), "mapping_logits batch mismatch");
    TORCH_CHECK(map_.size(1) == map_.size(2), "mapping_logits must be square");
    TORCH_CHECK(map_.size(1) > 0 && map_.size(1) <= 32, "S must be in [1,32]");
    TORCH_CHECK((out_.size(1) % map_.size(1)) == 0, "C must be divisible by S");

    TORCH_CHECK(sinkhorn_iter >= 0 && sinkhorn_iter <= 256, "sinkhorn_iter out of range");
    TORCH_CHECK(sinkhorn_eps >= 0.0, "sinkhorn_eps must be >= 0");
    TORCH_CHECK(sinkhorn_temperature > 0.0, "sinkhorn_temperature must be > 0");

    auto it_t  = npu_i32_scalar(sinkhorn_iter, out_);
    auto eps_t = npu_f32_scalar(sinkhorn_eps, out_);
    auto tmp_t = npu_f32_scalar(sinkhorn_temperature, out_);

    auto y = at::empty_like(out_);
    EXEC_NPU_CMD(aclnnMhcBlockBottleneck2dCustom,
                 out_, id_, map_,
                 it_t, eps_t, tmp_t,
                 y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("mhc_block_bottleneck2d_custom", &mhc_block_bottleneck2d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mhc_block_bottleneck2d_custom", &mhc_block_bottleneck2d_custom_impl_npu,
          "mhc_block_bottleneck2d_custom: sinkhorn + stream-mix + residual-add + relu (AscendC fp16)");
}
