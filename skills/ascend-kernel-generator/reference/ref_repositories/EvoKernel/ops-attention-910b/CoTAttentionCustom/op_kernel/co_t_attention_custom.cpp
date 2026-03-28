
#include "kernel_operator.h"
#include <cfloat>

class KernelCoTAttention {
public:
    __aicore__ inline KernelCoTAttention() {}

    __aicore__ inline void Init(GM_ADDR k1, GM_ADDR att, GM_ADDR v, GM_ADDR y,
                               uint32_t bs, uint32_t C, uint32_t H, uint32_t W,
                               uint32_t hw, uint32_t totalRows, uint32_t blockRows, uint32_t unroll)
    {
        this->bs = bs;
        this->C = C;
        this->H = H;
        this->W = W;
        this->hw = hw;
        this->totalRows = totalRows;
        this->blockRows = blockRows;
        this->unroll = unroll;

        const uint64_t k1Size = (uint64_t)bs * (uint64_t)C * (uint64_t)hw;
        const uint64_t rowMatSize = (uint64_t)totalRows * (uint64_t)hw;
        const uint64_t ySize  = k1Size;

        k1Gm.SetGlobalBuffer((__gm__ float*)k1, k1Size);
        attGm.SetGlobalBuffer((__gm__ float*)att, rowMatSize);
        vGm.SetGlobalBuffer((__gm__ float*)v, rowMatSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // Process 2 rows in parallel: per row need att[hw], v[hw], tmp[hw], out[hw], k1[hw]
        // We reuse buffers to minimize UB footprint: tmp used for exp/softmax weights.
        // Total floats = 2 * (att + v + tmp + out + k1) = 2*5*hw = 10*hw.
        pipe.InitBuffer(ubuf, (uint64_t)10 * (uint64_t)hw * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId  = (uint32_t)AscendC::GetBlockIdx();
        uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();
        if (coreNum == 0) return;

        uint32_t startRow = coreId * this->blockRows;
        uint32_t endRow = startRow + this->blockRows;
        if (endRow > this->totalRows) endRow = this->totalRows;
        if (startRow >= endRow) return;

        // Unrolled processing: two rows per iteration when possible.
        uint32_t row = startRow;
        for (; row + 1 < endRow; row += 2) {
            Compute2Rows(row);
        }
        if (row < endRow) {
            Compute1Row(row);
        }
    }

private:
    __aicore__ inline void SoftmaxMulAddOneRow(uint32_t row,
                                              AscendC::LocalTensor<float>& attL,
                                              AscendC::LocalTensor<float>& vL,
                                              AscendC::LocalTensor<float>& tmpL,
                                              AscendC::LocalTensor<float>& outL,
                                              AscendC::LocalTensor<float>& k1L)
    {
        const uint64_t rowOff = (uint64_t)row * (uint64_t)this->hw;

        // Load att and v and k1
        AscendC::DataCopy(attL, attGm[rowOff], this->hw);
        AscendC::DataCopy(vL,   vGm[rowOff],   this->hw);
        AscendC::DataCopy(k1L,  k1Gm[rowOff],  this->hw);
        AscendC::PipeBarrier<PIPE_V>();

        // Compute max (hierarchical reduction to reduce scalar loop length)
        // For hw=49 typical, scalar is small, but we still reduce barriers by doing math in-place.
        float maxVal = -FLT_MAX;
        for (uint32_t i = 0; i < this->hw; ++i) {
            float a = attL.GetValue(i);
            maxVal = (a > maxVal) ? a : maxVal;
        }

        // tmp = exp(att - max)
        AscendC::Adds(tmpL, attL, -maxVal, (int32_t)this->hw);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(tmpL, tmpL, (int32_t)this->hw);
        AscendC::PipeBarrier<PIPE_V>();

        float sumExp = 0.0f;
        for (uint32_t i = 0; i < this->hw; ++i) {
            sumExp += tmpL.GetValue(i);
        }
        float invSum = 1.0f / (sumExp > 0.0f ? sumExp : 1.0f);

        // tmp = softmax weights
        AscendC::Muls(tmpL, tmpL, invSum, (int32_t)this->hw);
        AscendC::PipeBarrier<PIPE_V>();

        // out = tmp * v
        AscendC::Mul(outL, tmpL, vL, (int32_t)this->hw);
        AscendC::PipeBarrier<PIPE_V>();

        // out = k1 + out
        AscendC::Add(outL, k1L, outL, (int32_t)this->hw);
        AscendC::PipeBarrier<PIPE_V>();

        // Store
        AscendC::DataCopy(yGm[rowOff], outL, this->hw);
    }

    __aicore__ inline void Compute2Rows(uint32_t row0)
    {
        AscendC::LocalTensor<float> base = ubuf.Get<float>();

        // Row0 tensors
        AscendC::LocalTensor<float> att0 = base;
        AscendC::LocalTensor<float> v0   = base[this->hw];
        AscendC::LocalTensor<float> tmp0 = base[2 * this->hw];
        AscendC::LocalTensor<float> out0 = base[3 * this->hw];
        AscendC::LocalTensor<float> k10  = base[4 * this->hw];

        // Row1 tensors
        AscendC::LocalTensor<float> att1 = base[5 * this->hw];
        AscendC::LocalTensor<float> v1   = base[6 * this->hw];
        AscendC::LocalTensor<float> tmp1 = base[7 * this->hw];
        AscendC::LocalTensor<float> out1 = base[8 * this->hw];
        AscendC::LocalTensor<float> k11  = base[9 * this->hw];

        // Interleave the two rows to increase ILP and reduce pipeline gaps.
        // Do row0 completely then row1; UB already holds both sets, barriers minimized inside helper.
        SoftmaxMulAddOneRow(row0, att0, v0, tmp0, out0, k10);
        AscendC::PipeBarrier<PIPE_V>();
        SoftmaxMulAddOneRow(row0 + 1, att1, v1, tmp1, out1, k11);
    }

    __aicore__ inline void Compute1Row(uint32_t row)
    {
        AscendC::LocalTensor<float> base = ubuf.Get<float>();
        AscendC::LocalTensor<float> attL = base;
        AscendC::LocalTensor<float> vL   = base[this->hw];
        AscendC::LocalTensor<float> tmpL = base[2 * this->hw];
        AscendC::LocalTensor<float> outL = base[3 * this->hw];
        AscendC::LocalTensor<float> k1L  = base[4 * this->hw];

        SoftmaxMulAddOneRow(row, attL, vL, tmpL, outL, k1L);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> ubuf;

    AscendC::GlobalTensor<float> k1Gm;
    AscendC::GlobalTensor<float> attGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t bs {0}, C {0}, H {0}, W {0}, hw {0};
    uint32_t totalRows {0}, blockRows {0}, unroll {2};
};

extern "C" __global__ __aicore__ void co_t_attention_custom(GM_ADDR k1, GM_ADDR att, GM_ADDR v, GM_ADDR y,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelCoTAttention op;
    op.Init(k1, att, v, y,
            tiling_data.bs, tiling_data.C, tiling_data.H, tiling_data.W,
            tiling_data.hw, tiling_data.totalRows, tiling_data.blockRows, tiling_data.unroll);
    op.Process();
}
