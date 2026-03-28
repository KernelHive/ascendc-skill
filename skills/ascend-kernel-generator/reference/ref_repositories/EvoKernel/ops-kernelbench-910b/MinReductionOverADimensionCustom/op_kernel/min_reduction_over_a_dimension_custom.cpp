
#include "kernel_operator.h"

class KernelMinReductionOverDim1 {
public:
    __aicore__ inline KernelMinReductionOverDim1() : pipe_(nullptr) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t batch, uint32_t reduceDim, uint32_t innerDim,
                               uint32_t outerCount, uint32_t colsPerCore,
                               AscendC::TPipe *pipe)
    {
        pipe_ = pipe;
        batch_ = batch;
        reduceDim_ = reduceDim;
        innerDim_ = innerDim;
        outerCount_ = outerCount;
        colsPerCore_ = colsPerCore;

        const uint64_t xSize = static_cast<uint64_t>(batch_) * static_cast<uint64_t>(reduceDim_) * static_cast<uint64_t>(innerDim_);
        const uint64_t ySize = static_cast<uint64_t>(batch_) * static_cast<uint64_t>(innerDim_);

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), xSize);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), ySize);

        // Persistent UB buffers (success pattern): contiguous slice + tmp scratch + 1-float output.
        pipe_->InitBuffer(bufSlice_, static_cast<uint32_t>(reduceDim_ * sizeof(float)));
        pipe_->InitBuffer(bufTmp_,   static_cast<uint32_t>(reduceDim_ * sizeof(float)));
        pipe_->InitBuffer(bufOut1_,  static_cast<uint32_t>(sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t start = coreId * colsPerCore_;
        uint32_t end = start + colsPerCore_;
        if (end > outerCount_) end = outerCount_;

        if (start >= end) {
            return;
        }

        AscendC::LocalTensor<float> sliceUb = bufSlice_.Get<float>();
        AscendC::LocalTensor<float> tmpUb   = bufTmp_.Get<float>();
        AscendC::LocalTensor<float> out1Ub  = bufOut1_.Get<float>();

        // Each output corresponds to (b, i) where:
        // input x shape: [batch, reduceDim, innerDim]
        // reduce over dim=1 => min over r for fixed (b, i).
        for (uint32_t col = start; col < end; ++col) {
            const uint32_t b = col / innerDim_;
            const uint32_t i = col - b * innerDim_;

            // Gather x[b, :, i] into contiguous UB slice (length reduceDim).
            // GM layout is contiguous in last dim; stepping r increases by innerDim_.
            AscendC::DataCopyParams p;
            p.blockCount = reduceDim_;                       // number of blocks
            p.blockLen = 1;                                  // one float per block
            p.srcStride = static_cast<uint16_t>((innerDim_ - 1) * sizeof(float)); // bytes between blocks (after reading 1)
            p.dstStride = 0;

            const uint64_t base = (static_cast<uint64_t>(b) * static_cast<uint64_t>(reduceDim_) * static_cast<uint64_t>(innerDim_)) +
                                  static_cast<uint64_t>(i);

            AscendC::DataCopy(sliceUb, xGm_[base], p);
            AscendC::PipeBarrier<PIPE_MTE2>();

            // ReduceMin over contiguous slice (reduceDim elements). No index needed.
            AscendC::ReduceMin<float>(out1Ub, sliceUb, tmpUb, static_cast<int32_t>(reduceDim_), false);
            AscendC::PipeBarrier<PIPE_V>();

            // Write scalar to GM.
            AscendC::DataCopy(yGm_[col], out1Ub, 1);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t batch_;
    uint32_t reduceDim_;
    uint32_t innerDim_;
    uint32_t outerCount_;
    uint32_t colsPerCore_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufSlice_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTmp_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOut1_;
};

extern "C" __global__ __aicore__ void min_reduction_over_a_dimension_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    KernelMinReductionOverDim1 op;
    op.Init(x, y,
            tilingData.batch,
            tilingData.reduceDim,
            tilingData.innerDim,
            tilingData.outerCount,
            tilingData.colsPerCore,
            &pipe);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
