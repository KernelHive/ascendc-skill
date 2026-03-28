
#include "kernel_operator.h"

class KernelVanillaRnnHiddenCustom {
public:
    __aicore__ inline KernelVanillaRnnHiddenCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR h0,
                               GM_ADDR w_i2h, GM_ADDR b_i2h,
                               GM_ADDR w_h2o, GM_ADDR b_h2o,
                               GM_ADDR y,
                               uint32_t T, uint32_t B, uint32_t I, uint32_t H, uint32_t O, uint32_t K,
                               uint32_t totalB, uint32_t blockB)
    {
        T_ = T; B_ = B; I_ = I; H_ = H; O_ = O; K_ = K;
        totalB_ = totalB;
        blockB_ = (blockB == 0 ? 1 : blockB);

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        h0Gm_.SetGlobalBuffer((__gm__ float*)h0);
        wi2hGm_.SetGlobalBuffer((__gm__ float*)w_i2h);
        bi2hGm_.SetGlobalBuffer((__gm__ float*)b_i2h);
        wh2oGm_.SetGlobalBuffer((__gm__ float*)w_h2o);
        bh2oGm_.SetGlobalBuffer((__gm__ float*)b_h2o);
        yGm_.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline void Process()
    {
        if (T_ == 0 || B_ == 0 || I_ == 0 || H_ == 0 || O_ == 0) return;

        AscendC::TPipe pipe;
        pipe.Init();

        // UB layout (float32), allocated once per block and reused:
        // comb(K) + hid(H) + pre(H) + tanh(H) + tmp(H) + out(O) + bi2h(H) + bh2o(O)
        // floats = K + 5H + 2O
        const uint32_t floatsNeed = K_ + 5U * H_ + 2U * O_;
        const uint32_t bytesNeed = floatsNeed * sizeof(float);

        AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, bytesNeed);

        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t startB = bid * blockB_;
        uint32_t endB = startB + blockB_;
        if (startB >= totalB_) return;
        if (endB > totalB_) endB = totalB_;

        for (uint32_t b = startB; b < endB; ++b) {
            ProcessOneB(buf, b);
        }
    }

private:
    __aicore__ inline void TanhVec(AscendC::LocalTensor<float>& dst,
                                  const AscendC::LocalTensor<float>& src,
                                  AscendC::LocalTensor<float>& tmp,
                                  int32_t count)
    {
        // tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
        AscendC::Muls(tmp, src, -2.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(tmp, tmp, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(dst, 1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(dst, dst, tmp, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(tmp, tmp, 1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(dst, dst, tmp, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessOneB(AscendC::TBuf<AscendC::TPosition::VECCALC>& buf, uint32_t b)
    {
        AscendC::LocalTensor<float> ub = buf.Get<float>();
        uint32_t off = 0;

        AscendC::LocalTensor<float> comb  = ub[off]; off += K_;
        AscendC::LocalTensor<float> hid   = ub[off]; off += H_;
        AscendC::LocalTensor<float> pre   = ub[off]; off += H_;
        AscendC::LocalTensor<float> tanh  = ub[off]; off += H_;
        AscendC::LocalTensor<float> tmp   = ub[off]; off += H_;   // MUST be H-sized scratch (avoid prior OOB)
        AscendC::LocalTensor<float> out   = ub[off]; off += O_;
        AscendC::LocalTensor<float> biUb  = ub[off]; off += H_;
        AscendC::LocalTensor<float> boUb  = ub[off];

        // Cache biases once per b
        AscendC::DataCopy(biUb, bi2hGm_[0], H_);
        AscendC::DataCopy(boUb, bh2oGm_[0], O_);
        AscendC::PipeBarrier<PIPE_MTE2>();

        // hid = h0[b,:]
        const uint64_t h0Base = (uint64_t)b * (uint64_t)H_;
        AscendC::DataCopy(hid, h0Gm_[h0Base], H_);
        AscendC::PipeBarrier<PIPE_MTE2>();

        for (uint32_t t = 0; t < T_; ++t) {
            // comb[0:I] = x[t,b,:] (bulk)
            const uint64_t xBase = ((uint64_t)t * (uint64_t)B_ + (uint64_t)b) * (uint64_t)I_;
            AscendC::DataCopy(comb, xGm_[xBase], I_);
            AscendC::PipeBarrier<PIPE_MTE2>();

            // comb[I:I+H] = hid (UB->UB)
            AscendC::DataCopy(comb[I_], hid, H_);
            AscendC::PipeBarrier<PIPE_V>();

            // pre = bi2h (UB->UB)
            AscendC::DataCopy(pre, biUb, H_);
            AscendC::PipeBarrier<PIPE_V>();

            // pre[j] += dot(w_i2h[j,:], comb[:])  (scalar accumulate; operands in UB/GM)
            for (uint32_t j = 0; j < H_; ++j) {
                float acc = pre.GetValue(j);
                const uint64_t rowBase = (uint64_t)j * (uint64_t)K_;
                // Unroll by 4 to reduce loop overhead somewhat without extra UB.
                uint32_t k = 0;
                for (; k + 3 < K_; k += 4) {
                    float c0 = comb.GetValue(k);
                    float c1 = comb.GetValue(k + 1);
                    float c2 = comb.GetValue(k + 2);
                    float c3 = comb.GetValue(k + 3);
                    acc += c0 * wi2hGm_.GetValue(rowBase + (uint64_t)k);
                    acc += c1 * wi2hGm_.GetValue(rowBase + (uint64_t)k + 1ULL);
                    acc += c2 * wi2hGm_.GetValue(rowBase + (uint64_t)k + 2ULL);
                    acc += c3 * wi2hGm_.GetValue(rowBase + (uint64_t)k + 3ULL);
                }
                for (; k < K_; ++k) {
                    acc += comb.GetValue(k) * wi2hGm_.GetValue(rowBase + (uint64_t)k);
                }
                pre.SetValue(j, acc);
            }

            // hid = tanh(pre) (vector)
            TanhVec(tanh, pre, tmp, (int32_t)H_);
            AscendC::DataCopy(hid, tanh, H_);
            AscendC::PipeBarrier<PIPE_V>();

            // out = bh2o (UB->UB)
            AscendC::DataCopy(out, boUb, O_);
            AscendC::PipeBarrier<PIPE_V>();

            // out[o] += dot(w_h2o[o,:], hid)
            for (uint32_t o = 0; o < O_; ++o) {
                float acc = out.GetValue(o);
                const uint64_t rowBase = (uint64_t)o * (uint64_t)H_;
                uint32_t j0 = 0;
                for (; j0 + 3 < H_; j0 += 4) {
                    float h0v = hid.GetValue(j0);
                    float h1v = hid.GetValue(j0 + 1);
                    float h2v = hid.GetValue(j0 + 2);
                    float h3v = hid.GetValue(j0 + 3);
                    acc += h0v * wh2oGm_.GetValue(rowBase + (uint64_t)j0);
                    acc += h1v * wh2oGm_.GetValue(rowBase + (uint64_t)j0 + 1ULL);
                    acc += h2v * wh2oGm_.GetValue(rowBase + (uint64_t)j0 + 2ULL);
                    acc += h3v * wh2oGm_.GetValue(rowBase + (uint64_t)j0 + 3ULL);
                }
                for (; j0 < H_; ++j0) {
                    acc += hid.GetValue(j0) * wh2oGm_.GetValue(rowBase + (uint64_t)j0);
                }
                out.SetValue(o, acc);
            }

            // store y[t,b,:] (bulk)
            const uint64_t yBase = ((uint64_t)t * (uint64_t)B_ + (uint64_t)b) * (uint64_t)O_;
            AscendC::DataCopy(yGm_[yBase], out, O_);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    AscendC::GlobalTensor<float> xGm_, h0Gm_, wi2hGm_, bi2hGm_, wh2oGm_, bh2oGm_, yGm_;
    uint32_t T_{0}, B_{0}, I_{0}, H_{0}, O_{0}, K_{0};
    uint32_t totalB_{0}, blockB_{1};
};

extern "C" __global__ __aicore__ void vanilla_rnn_hidden_custom(
    GM_ADDR x, GM_ADDR h0,
    GM_ADDR w_i2h, GM_ADDR b_i2h,
    GM_ADDR w_h2o, GM_ADDR b_h2o,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelVanillaRnnHiddenCustom op;
    op.Init(x, h0, w_i2h, b_i2h, w_h2o, b_h2o, y,
            td.T, td.B, td.I, td.H, td.O, td.K,
            td.totalB, td.blockB);
    op.Process();
}
