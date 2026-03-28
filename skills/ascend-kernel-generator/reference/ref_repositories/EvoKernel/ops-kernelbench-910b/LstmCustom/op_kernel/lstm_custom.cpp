
#include "kernel_operator.h"

class KernelLstmCustom {
public:
    __aicore__ inline KernelLstmCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR h0, GM_ADDR c0,
                               GM_ADDR w_ih, GM_ADDR w_hh,
                               GM_ADDR b_ih, GM_ADDR b_hh,
                               GM_ADDR fc_w, GM_ADDR fc_b,
                               GM_ADDR y,
                               uint32_t B, uint32_t S, uint32_t I, uint32_t H, uint32_t L, uint32_t O,
                               uint32_t totalB, uint32_t blockB)
    {
        B_ = B; S_ = S; I_ = I; H_ = H; L_ = L; O_ = O;
        totalB_ = totalB;
        blockB_ = (blockB == 0 ? 1 : blockB);

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        h0Gm_.SetGlobalBuffer((__gm__ float*)h0);
        c0Gm_.SetGlobalBuffer((__gm__ float*)c0);
        wihGm_.SetGlobalBuffer((__gm__ float*)w_ih);
        whhGm_.SetGlobalBuffer((__gm__ float*)w_hh);
        bihGm_.SetGlobalBuffer((__gm__ float*)b_ih);
        bhhGm_.SetGlobalBuffer((__gm__ float*)b_hh);
        fcWGm_.SetGlobalBuffer((__gm__ float*)fc_w);
        fcBGm_.SetGlobalBuffer((__gm__ float*)fc_b);
        yGm_.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline void Process()
    {
        if (B_ == 0 || S_ == 0 || H_ == 0 || L_ == 0 || O_ == 0) return;

        AscendC::TPipe pipe;
        pipe.Init();

        // UB scratch:
        // hstack(L*H) + cstack(L*H) +
        // layerIn(H) + hcur(H) +
        // pre gates (4H) + act gates (4H) +
        // tanh_c(H) + tmp(H) +
        // lastH(H) + fcTmp(O)
        const uint32_t floatsNeed =
            (2U * L_ * H_) +
            (2U * H_) +
            (8U * H_) +
            (2U * H_) +
            (1U * H_) +
            (1U * O_);
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
    __aicore__ inline void Sigmoid(AscendC::LocalTensor<float>& dst,
                                  const AscendC::LocalTensor<float>& src,
                                  int32_t count)
    {
        AscendC::Muls(dst, src, -1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(dst, dst, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(dst, dst, 1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Reciprocal(dst, dst, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TanhApprox(AscendC::LocalTensor<float>& dst,
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

        AscendC::LocalTensor<float> hstack  = ub;                      // L*H
        AscendC::LocalTensor<float> cstack  = ub[L_ * H_];             // L*H
        AscendC::LocalTensor<float> work    = ub[2U * L_ * H_];

        AscendC::LocalTensor<float> layerIn = work;                    // H
        AscendC::LocalTensor<float> hcur    = work[1U * H_];           // H

        AscendC::LocalTensor<float> pre_i   = work[2U * H_];           // H
        AscendC::LocalTensor<float> pre_f   = work[3U * H_];
        AscendC::LocalTensor<float> pre_g   = work[4U * H_];
        AscendC::LocalTensor<float> pre_o   = work[5U * H_];

        AscendC::LocalTensor<float> act_i   = work[6U * H_];           // H
        AscendC::LocalTensor<float> act_f   = work[7U * H_];
        AscendC::LocalTensor<float> act_g   = work[8U * H_];
        AscendC::LocalTensor<float> act_o   = work[9U * H_];

        AscendC::LocalTensor<float> tanh_c  = work[10U * H_];          // H
        AscendC::LocalTensor<float> tmp     = work[11U * H_];          // H

        AscendC::LocalTensor<float> lastH   = work[12U * H_];          // H
        AscendC::LocalTensor<float> fcTmp   = work[13U * H_];          // O (O<=H)

        // init h/c stacks from h0/c0 for batch b (keep scalar loads to avoid any potential copy-path quirks)
        for (uint32_t ll = 0; ll < L_; ++ll) {
            const uint64_t base = ((uint64_t)ll * (uint64_t)B_ + (uint64_t)b) * (uint64_t)H_;
            for (uint32_t k = 0; k < H_; ++k) {
                hstack.SetValue(ll * H_ + k, h0Gm_.GetValue(base + (uint64_t)k));
                cstack.SetValue(ll * H_ + k, c0Gm_.GetValue(base + (uint64_t)k));
            }
        }

        // time loop
        for (uint32_t t = 0; t < S_; ++t) {

            // layers
            for (uint32_t ll = 0; ll < L_; ++ll) {
                // Prepare layer input:
                // ll==0: read x[b,t,:I] on the fly (no padding vector build)
                // ll>0 : use previous layer output stored in layerIn (H values)
                if (ll > 0U) {
                    // layerIn already contains previous layer output from last iteration
                }

                // hcur = hstack[ll]
                for (uint32_t k = 0; k < H_; ++k) {
                    hcur.SetValue(k, hstack.GetValue(ll * H_ + k));
                }

                const uint64_t gateBase = (uint64_t)ll * 4ULL * (uint64_t)H_;

                // init pre gates with bias sum (kept identical to baseline: scalar GM reads)
                for (uint32_t h = 0; h < H_; ++h) {
                    const uint64_t offI = gateBase + 0ULL * (uint64_t)H_ + (uint64_t)h;
                    const uint64_t offF = gateBase + 1ULL * (uint64_t)H_ + (uint64_t)h;
                    const uint64_t offG = gateBase + 2ULL * (uint64_t)H_ + (uint64_t)h;
                    const uint64_t offO = gateBase + 3ULL * (uint64_t)H_ + (uint64_t)h;

                    pre_i.SetValue(h, bihGm_.GetValue(offI) + bhhGm_.GetValue(offI));
                    pre_f.SetValue(h, bihGm_.GetValue(offF) + bhhGm_.GetValue(offF));
                    pre_g.SetValue(h, bihGm_.GetValue(offG) + bhhGm_.GetValue(offG));
                    pre_o.SetValue(h, bihGm_.GetValue(offO) + bhhGm_.GetValue(offO));
                }

                // gate GEMV: compute 4 gates together to reduce loop/control overhead and reuse x/h scalars
                // x length depends on layer:
                const uint32_t xLen = (ll == 0U ? I_ : H_);
                const uint64_t xBase = ((uint64_t)b * (uint64_t)S_ + (uint64_t)t) * (uint64_t)I_;

                for (uint32_t h_out = 0; h_out < H_; ++h_out) {
                    const uint64_t rowI = (gateBase + 0ULL * (uint64_t)H_ + (uint64_t)h_out) * (uint64_t)H_;
                    const uint64_t rowF = (gateBase + 1ULL * (uint64_t)H_ + (uint64_t)h_out) * (uint64_t)H_;
                    const uint64_t rowG = (gateBase + 2ULL * (uint64_t)H_ + (uint64_t)h_out) * (uint64_t)H_;
                    const uint64_t rowO = (gateBase + 3ULL * (uint64_t)H_ + (uint64_t)h_out) * (uint64_t)H_;

                    float accI = pre_i.GetValue(h_out);
                    float accF = pre_f.GetValue(h_out);
                    float accG = pre_g.GetValue(h_out);
                    float accO = pre_o.GetValue(h_out);

                    // x contribution
                    if (ll == 0U) {
                        for (uint32_t k = 0; k < xLen; ++k) {
                            const float xv = xGm_.GetValue(xBase + (uint64_t)k);
                            const uint64_t kk = (uint64_t)k;
                            accI += xv * wihGm_.GetValue(rowI + kk);
                            accF += xv * wihGm_.GetValue(rowF + kk);
                            accG += xv * wihGm_.GetValue(rowG + kk);
                            accO += xv * wihGm_.GetValue(rowO + kk);
                        }
                    } else {
                        for (uint32_t k = 0; k < xLen; ++k) {
                            const float xv = layerIn.GetValue(k);
                            const uint64_t kk = (uint64_t)k;
                            accI += xv * wihGm_.GetValue(rowI + kk);
                            accF += xv * wihGm_.GetValue(rowF + kk);
                            accG += xv * wihGm_.GetValue(rowG + kk);
                            accO += xv * wihGm_.GetValue(rowO + kk);
                        }
                    }

                    // h contribution
                    for (uint32_t k = 0; k < H_; ++k) {
                        const float hv = hcur.GetValue(k);
                        const uint64_t kk = (uint64_t)k;
                        accI += hv * whhGm_.GetValue(rowI + kk);
                        accF += hv * whhGm_.GetValue(rowF + kk);
                        accG += hv * whhGm_.GetValue(rowG + kk);
                        accO += hv * whhGm_.GetValue(rowO + kk);
                    }

                    pre_i.SetValue(h_out, accI);
                    pre_f.SetValue(h_out, accF);
                    pre_g.SetValue(h_out, accG);
                    pre_o.SetValue(h_out, accO);
                }

                // activations
                Sigmoid(act_i, pre_i, (int32_t)H_);
                Sigmoid(act_f, pre_f, (int32_t)H_);
                Sigmoid(act_o, pre_o, (int32_t)H_);
                TanhApprox(act_g, pre_g, tmp, (int32_t)H_);

                // c = f*c + i*g
                for (uint32_t k = 0; k < H_; ++k) {
                    float cprev = cstack.GetValue(ll * H_ + k);
                    float cnew = act_f.GetValue(k) * cprev + act_i.GetValue(k) * act_g.GetValue(k);
                    cstack.SetValue(ll * H_ + k, cnew);
                }

                // h = o * tanh(c)
                for (uint32_t k = 0; k < H_; ++k) {
                    tmp.SetValue(k, cstack.GetValue(ll * H_ + k));
                }
                TanhApprox(tanh_c, tmp, pre_i /*reuse*/, (int32_t)H_);
                for (uint32_t k = 0; k < H_; ++k) {
                    float hnew = act_o.GetValue(k) * tanh_c.GetValue(k);
                    hstack.SetValue(ll * H_ + k, hnew);
                    layerIn.SetValue(k, hnew); // next layer input
                }
            }

            if (t == S_ - 1U) {
                for (uint32_t k = 0; k < H_; ++k) lastH.SetValue(k, layerIn.GetValue(k));
            }
        }

        // FC: y[b,o] = dot(lastH, fc_w[o,:]) + fc_b[o]
        const uint64_t yBase = (uint64_t)b * (uint64_t)O_;
        for (uint32_t o = 0; o < O_; ++o) {
            const uint64_t wRow = (uint64_t)o * (uint64_t)H_;
            float acc = fcBGm_.GetValue((uint64_t)o);
            for (uint32_t k = 0; k < H_; ++k) {
                acc += lastH.GetValue(k) * fcWGm_.GetValue(wRow + (uint64_t)k);
            }
            fcTmp.SetValue(o, acc);
        }
        for (uint32_t o = 0; o < O_; ++o) {
            yGm_.SetValue(yBase + (uint64_t)o, fcTmp.GetValue(o));
        }
    }

private:
    AscendC::GlobalTensor<float> xGm_, h0Gm_, c0Gm_, wihGm_, whhGm_, bihGm_, bhhGm_, fcWGm_, fcBGm_, yGm_;
    uint32_t B_{0}, S_{0}, I_{0}, H_{0}, L_{0}, O_{0};
    uint32_t totalB_{0}, blockB_{1};
};

extern "C" __global__ __aicore__ void lstm_custom(
    GM_ADDR x, GM_ADDR h0, GM_ADDR c0,
    GM_ADDR w_ih, GM_ADDR w_hh, GM_ADDR b_ih, GM_ADDR b_hh,
    GM_ADDR fc_w, GM_ADDR fc_b,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelLstmCustom op;
    op.Init(x, h0, c0, w_ih, w_hh, b_ih, b_hh, fc_w, fc_b, y,
            td.B, td.S, td.I, td.H, td.L, td.O,
            td.totalB, td.blockB);
    op.Process();
}
