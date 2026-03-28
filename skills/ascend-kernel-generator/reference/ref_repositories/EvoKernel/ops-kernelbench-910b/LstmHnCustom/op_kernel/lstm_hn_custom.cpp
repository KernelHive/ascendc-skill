
#include "kernel_operator.h"

class KernelLstmHnCustom {
public:
    __aicore__ inline KernelLstmHnCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR h0, GM_ADDR c0,
                               GM_ADDR w_ih, GM_ADDR w_hh,
                               GM_ADDR b_ih, GM_ADDR b_hh,
                               GM_ADDR h_n,
                               uint32_t B, uint32_t S, uint32_t I, uint32_t H, uint32_t L,
                               uint32_t totalB, uint32_t blockB)
    {
        B_ = B; S_ = S; I_ = I; H_ = H; L_ = L;
        totalB_ = totalB;
        blockB_ = (blockB == 0 ? 1 : blockB);

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        h0Gm_.SetGlobalBuffer((__gm__ float*)h0);
        c0Gm_.SetGlobalBuffer((__gm__ float*)c0);
        wihGm_.SetGlobalBuffer((__gm__ float*)w_ih);
        whhGm_.SetGlobalBuffer((__gm__ float*)w_hh);
        bihGm_.SetGlobalBuffer((__gm__ float*)b_ih);
        bhhGm_.SetGlobalBuffer((__gm__ float*)b_hh);
        hnGm_.SetGlobalBuffer((__gm__ float*)h_n);
    }

    __aicore__ inline void Process()
    {
        if (B_ == 0 || S_ == 0 || H_ == 0 || L_ == 0) return;

        AscendC::TPipe pipe;
        pipe.Init();

        // UB scratch (float32):
        // hstack(L*H) + cstack(L*H)
        // xpad(H) + hcur(H)
        // pre_i/f/g/o (4H) + act_i/f/g/o (4H)
        // tanh_c(H) + tmp(H)
        const uint32_t floatsNeed = (2U * L_ * H_) + (2U * H_) + (8U * H_) + (2U * H_);
        AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, floatsNeed * sizeof(float));

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
    __aicore__ inline void SigmoidInplace(AscendC::LocalTensor<float>& x, int32_t count)
    {
        AscendC::Muls(x, x, -1.0f, count);
        AscendC::Exp(x, x, count);
        AscendC::Adds(x, x, 1.0f, count);
        AscendC::Reciprocal(x, x, count);
    }

    __aicore__ inline void TanhApprox(AscendC::LocalTensor<float>& dst,
                                     const AscendC::LocalTensor<float>& src,
                                     AscendC::LocalTensor<float>& tmp,
                                     int32_t count)
    {
        AscendC::Muls(tmp, src, -2.0f, count);
        AscendC::Exp(tmp, tmp, count);
        AscendC::Duplicate(dst, 1.0f, count);
        AscendC::Sub(dst, dst, tmp, count);
        AscendC::Adds(tmp, tmp, 1.0f, count);
        AscendC::Div(dst, dst, tmp, count);
    }

    __aicore__ inline void ProcessOneB(AscendC::TBuf<AscendC::TPosition::VECCALC>& buf, uint32_t b)
    {
        AscendC::LocalTensor<float> ub = buf.Get<float>();

        AscendC::LocalTensor<float> hstack = ub;                      // L*H
        AscendC::LocalTensor<float> cstack = ub[L_ * H_];             // L*H
        AscendC::LocalTensor<float> work   = ub[2U * L_ * H_];

        AscendC::LocalTensor<float> xpad   = work;                    // H
        AscendC::LocalTensor<float> hcur   = work[1U * H_];           // H

        AscendC::LocalTensor<float> pre_i  = work[2U * H_];           // H
        AscendC::LocalTensor<float> pre_f  = work[3U * H_];
        AscendC::LocalTensor<float> pre_g  = work[4U * H_];
        AscendC::LocalTensor<float> pre_o  = work[5U * H_];

        AscendC::LocalTensor<float> act_i  = work[6U * H_];           // H
        AscendC::LocalTensor<float> act_f  = work[7U * H_];
        AscendC::LocalTensor<float> act_g  = work[8U * H_];
        AscendC::LocalTensor<float> act_o  = work[9U * H_];

        AscendC::LocalTensor<float> tanh_c = work[10U * H_];          // H
        AscendC::LocalTensor<float> tmp    = work[11U * H_];          // H

        // init h/c stacks from h0/c0 for batch b using MTE (contiguous H)
        for (uint32_t ll = 0; ll < L_; ++ll) {
            const uint64_t gmBase = ((uint64_t)ll * (uint64_t)B_ + (uint64_t)b) * (uint64_t)H_;
            AscendC::DataCopy(hstack[ll * H_], h0Gm_[gmBase], H_);
            AscendC::DataCopy(cstack[ll * H_], c0Gm_[gmBase], H_);
        }

        for (uint32_t t = 0; t < S_; ++t) {
            // xpad = [x[b,t,:I], zeros(H-I)] using MTE + vector tail zeroing
            const uint64_t xBase = ((uint64_t)b * (uint64_t)S_ + (uint64_t)t) * (uint64_t)I_;
            AscendC::DataCopy(xpad, xGm_[xBase], I_);
            if (H_ > I_) {
                AscendC::Duplicate(xpad[I_], 0.0f, (int32_t)(H_ - I_));
            }

            for (uint32_t ll = 0; ll < L_; ++ll) {
                const uint64_t gateBase = (uint64_t)ll * 4ULL * (uint64_t)H_;
                const uint32_t hOff = ll * H_;

                // hcur = hstack[ll] via MTE (UB->UB DataCopy is supported and cheaper than scalar loop)
                AscendC::DataCopy(hcur, hstack[hOff], H_);

                // init pre gates from (b_ih + b_hh) (kept scalar but with hoisted offsets)
                const uint64_t offI0 = gateBase + 0ULL * (uint64_t)H_;
                const uint64_t offF0 = gateBase + 1ULL * (uint64_t)H_;
                const uint64_t offG0 = gateBase + 2ULL * (uint64_t)H_;
                const uint64_t offO0 = gateBase + 3ULL * (uint64_t)H_;
                for (uint32_t h = 0; h < H_; ++h) {
                    const uint64_t hh = (uint64_t)h;
                    pre_i.SetValue(h, bihGm_.GetValue(offI0 + hh) + bhhGm_.GetValue(offI0 + hh));
                    pre_f.SetValue(h, bihGm_.GetValue(offF0 + hh) + bhhGm_.GetValue(offF0 + hh));
                    pre_g.SetValue(h, bihGm_.GetValue(offG0 + hh) + bhhGm_.GetValue(offG0 + hh));
                    pre_o.SetValue(h, bihGm_.GetValue(offO0 + hh) + bhhGm_.GetValue(offO0 + hh));
                }

                // GEMV per gate row, reuse hcur (avoid reading hstack with extra address arithmetic)
                for (uint32_t h_out = 0; h_out < H_; ++h_out) {
                    const uint64_t rowBase = (gateBase + (uint64_t)h_out) * (uint64_t)H_;
                    const uint64_t rI = rowBase + 0ULL * (uint64_t)H_ * (uint64_t)H_;
                    const uint64_t rF = rowBase + 1ULL * (uint64_t)H_ * (uint64_t)H_;
                    const uint64_t rG = rowBase + 2ULL * (uint64_t)H_ * (uint64_t)H_;
                    const uint64_t rO = rowBase + 3ULL * (uint64_t)H_ * (uint64_t)H_;

                    float accI = pre_i.GetValue(h_out);
                    float accF = pre_f.GetValue(h_out);
                    float accG = pre_g.GetValue(h_out);
                    float accO = pre_o.GetValue(h_out);

                    for (uint32_t k = 0; k < H_; ++k) {
                        const float xv = xpad.GetValue(k);
                        const float hv = hcur.GetValue(k);
                        const uint64_t kk = (uint64_t)k;

                        accI += xv * wihGm_.GetValue(rI + kk) + hv * whhGm_.GetValue(rI + kk);
                        accF += xv * wihGm_.GetValue(rF + kk) + hv * whhGm_.GetValue(rF + kk);
                        accG += xv * wihGm_.GetValue(rG + kk) + hv * whhGm_.GetValue(rG + kk);
                        accO += xv * wihGm_.GetValue(rO + kk) + hv * whhGm_.GetValue(rO + kk);
                    }

                    pre_i.SetValue(h_out, accI);
                    pre_f.SetValue(h_out, accF);
                    pre_g.SetValue(h_out, accG);
                    pre_o.SetValue(h_out, accO);
                }

                AscendC::DataCopy(act_i, pre_i, H_);
                AscendC::DataCopy(act_f, pre_f, H_);
                AscendC::DataCopy(act_o, pre_o, H_);
                SigmoidInplace(act_i, (int32_t)H_);
                SigmoidInplace(act_f, (int32_t)H_);
                SigmoidInplace(act_o, (int32_t)H_);
                TanhApprox(act_g, pre_g, tmp, (int32_t)H_);

                // c = f*c + i*g
                for (uint32_t k = 0; k < H_; ++k) {
                    float cprev = cstack.GetValue(hOff + k);
                    float cnew = act_f.GetValue(k) * cprev + act_i.GetValue(k) * act_g.GetValue(k);
                    cstack.SetValue(hOff + k, cnew);
                }

                // h = o * tanh(c); becomes next-layer xpad
                AscendC::DataCopy(tmp, cstack[hOff], H_);
                TanhApprox(tanh_c, tmp, pre_i /*scratch*/, (int32_t)H_);
                for (uint32_t k = 0; k < H_; ++k) {
                    float hnew = act_o.GetValue(k) * tanh_c.GetValue(k);
                    hstack.SetValue(hOff + k, hnew);
                    xpad.SetValue(k, hnew);
                }
            }
        }

        // store h_n [L,B,H] for this batch using MTE
        for (uint32_t ll = 0; ll < L_; ++ll) {
            const uint64_t outBase = ((uint64_t)ll * (uint64_t)B_ + (uint64_t)b) * (uint64_t)H_;
            AscendC::DataCopy(hnGm_[outBase], hstack[ll * H_], H_);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm_, h0Gm_, c0Gm_, wihGm_, whhGm_, bihGm_, bhhGm_, hnGm_;
    uint32_t B_{0}, S_{0}, I_{0}, H_{0}, L_{0};
    uint32_t totalB_{0}, blockB_{1};
};

extern "C" __global__ __aicore__ void lstm_hn_custom(
    GM_ADDR x, GM_ADDR h0, GM_ADDR c0,
    GM_ADDR w_ih, GM_ADDR w_hh, GM_ADDR b_ih, GM_ADDR b_hh,
    GM_ADDR h_n, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelLstmHnCustom op;
    op.Init(x, h0, c0, w_ih, w_hh, b_ih, b_hh, h_n,
            td.B, td.S, td.I, td.H, td.L,
            td.totalB, td.blockB);
    op.Process();
}
