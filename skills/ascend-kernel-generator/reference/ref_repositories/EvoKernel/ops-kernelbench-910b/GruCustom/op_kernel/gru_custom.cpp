
#include "kernel_operator.h"

// GRU forward output-only (last layer outputs), specialized to:
// T=512, B=10, I=128, H=256, L=6, bias=True, batch_first=False, bidirectional=False
//
// Packed params expected (PyTorch GRU gate order: r,z,n):
//   w_ih: [L*3H, H]  (layer0 weight_ih [3H,I] placed into [:,:I] and zero padded to H columns)
//   w_hh: [L*3H, H]
//   b_ih: [L*3H]
//   b_hh: [L*3H]
//
// Output:
//   y: [T,B,H] float16, last layer outputs for each timestep.
//
// NOTE: Avoids any libm symbols; uses AscendC vector Exp/Reciprocal for sigmoid/tanh.

class KernelGruCustom {
public:
    __aicore__ inline KernelGruCustom() {}

    static constexpr uint32_t T = 512;
    static constexpr uint32_t B = 10;
    static constexpr uint32_t I = 128;
    static constexpr uint32_t H = 256;
    static constexpr uint32_t L = 6;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR h0,
                               GM_ADDR w_ih, GM_ADDR w_hh,
                               GM_ADDR b_ih, GM_ADDR b_hh,
                               GM_ADDR y)
    {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
        h0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(h0));
        wihGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w_ih));
        whhGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w_hh));
        bihGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b_ih));
        bhhGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b_hh));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));

        // UB arena (float):
        // hstack: L*H = 1536
        // xbuf0/xbuf1: 2*H = 512
        // hcur: H = 256
        // rpre,zpre,nxpre,nhpre: 4*H = 1024
        // r,z,npre,n: 4*H = 1024
        // tmp1,tmp2: 2*H = 512
        // Total = 4864 floats = 19KB
        pipe.InitBuffer(calcBuf, 20480);
    }

    __aicore__ inline void SigmoidInplace(AscendC::LocalTensor<float>& x, int32_t count)
    {
        // x = 1/(1+exp(-x))
        AscendC::Muls(x, x, -1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(x, x, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(x, x, 1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Reciprocal(x, x, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TanhOut(AscendC::LocalTensor<float>& dst,
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

    __aicore__ inline void ProcessBatch(uint32_t b)
    {
        AscendC::LocalTensor<float> ub = calcBuf.Get<float>();

        AscendC::LocalTensor<float> hstack = ub;                    // L*H
        uint32_t off = L * H;

        AscendC::LocalTensor<float> xbuf0  = ub[off];               off += H;
        AscendC::LocalTensor<float> xbuf1  = ub[off];               off += H;
        AscendC::LocalTensor<float> hcur   = ub[off];               off += H;

        AscendC::LocalTensor<float> rpre   = ub[off];               off += H;
        AscendC::LocalTensor<float> zpre   = ub[off];               off += H;
        AscendC::LocalTensor<float> nxpre  = ub[off];               off += H;
        AscendC::LocalTensor<float> nhpre  = ub[off];               off += H;

        AscendC::LocalTensor<float> r      = ub[off];               off += H;
        AscendC::LocalTensor<float> z      = ub[off];               off += H;
        AscendC::LocalTensor<float> npre   = ub[off];               off += H;
        AscendC::LocalTensor<float> n      = ub[off];               off += H;

        AscendC::LocalTensor<float> tmp1   = ub[off];               off += H; // scratch
        AscendC::LocalTensor<float> tmp2   = ub[off];               off += H; // scratch

        // Initialize hstack from h0.
        for (uint32_t ll = 0; ll < L; ++ll) {
            const uint64_t base = (static_cast<uint64_t>(ll) * B + b) * H;
            const uint32_t dstBase = ll * H;
            for (uint32_t k = 0; k < H; ++k) {
                hstack.SetValue(dstBase + k, static_cast<float>(h0Gm.GetValue(base + k)));
            }
        }

        for (uint32_t t = 0; t < T; ++t) {
            // layer0 input in xbuf0: x[t,b,:I] padded to H
            AscendC::Duplicate(xbuf0, 0.0f, H);
            AscendC::PipeBarrier<PIPE_V>();
            const uint64_t xBase = (static_cast<uint64_t>(t) * B + b) * I;
            for (uint32_t k = 0; k < I; ++k) {
                xbuf0.SetValue(k, static_cast<float>(xGm.GetValue(xBase + k)));
            }

            AscendC::LocalTensor<float> xin = xbuf0;
            AscendC::LocalTensor<float> xout = xbuf1;

            for (uint32_t ll = 0; ll < L; ++ll) {
                // Load current hidden into hcur
                const uint32_t hsOff = ll * H;
                for (uint32_t k = 0; k < H; ++k) {
                    hcur.SetValue(k, hstack.GetValue(hsOff + k));
                }

                // Bias init:
                // rpre = b_ir + b_hr
                // zpre = b_iz + b_hz
                // nxpre = b_in
                // nhpre = b_hn
                const uint64_t gateBase = static_cast<uint64_t>(ll) * 3ULL * H;
                for (uint32_t h = 0; h < H; ++h) {
                    const uint64_t offR = gateBase + 0ULL * H + h;
                    const uint64_t offZ = gateBase + 1ULL * H + h;
                    const uint64_t offN = gateBase + 2ULL * H + h;

                    const float bir = static_cast<float>(bihGm.GetValue(offR));
                    const float bhr = static_cast<float>(bhhGm.GetValue(offR));
                    const float biz = static_cast<float>(bihGm.GetValue(offZ));
                    const float bhz = static_cast<float>(bhhGm.GetValue(offZ));
                    const float bin = static_cast<float>(bihGm.GetValue(offN));
                    const float bhn = static_cast<float>(bhhGm.GetValue(offN));

                    rpre.SetValue(h, bir + bhr);
                    zpre.SetValue(h, biz + bhz);
                    nxpre.SetValue(h, bin);
                    nhpre.SetValue(h, bhn);
                }

                // GEMV for all 3 gates using the "simple" formulation (matches prior successful h_n kernel):
                // rpre/zpre/nxpre += W_ih * x
                // rpre/zpre/nhpre += W_hh * h
                const uint64_t rowBaseR = (gateBase + 0ULL * H) * static_cast<uint64_t>(H);
                const uint64_t rowBaseZ = (gateBase + 1ULL * H) * static_cast<uint64_t>(H);
                const uint64_t rowBaseN = (gateBase + 2ULL * H) * static_cast<uint64_t>(H);

                for (uint32_t h_out = 0; h_out < H; ++h_out) {
                    float rsum  = rpre.GetValue(h_out);
                    float zsum  = zpre.GetValue(h_out);
                    float nxsum = nxpre.GetValue(h_out);
                    float nhsum = nhpre.GetValue(h_out);

                    const uint64_t rowOff = static_cast<uint64_t>(h_out) * H;
                    const uint64_t rowIr = rowBaseR + rowOff;
                    const uint64_t rowIz = rowBaseZ + rowOff;
                    const uint64_t rowIn = rowBaseN + rowOff;

                    for (uint32_t k = 0; k < H; ++k) {
                        const float xv = xin.GetValue(k);
                        rsum  += xv * static_cast<float>(wihGm.GetValue(rowIr + k));
                        zsum  += xv * static_cast<float>(wihGm.GetValue(rowIz + k));
                        nxsum += xv * static_cast<float>(wihGm.GetValue(rowIn + k));
                    }

                    for (uint32_t k = 0; k < H; ++k) {
                        const float hv = hcur.GetValue(k);
                        rsum  += hv * static_cast<float>(whhGm.GetValue(rowIr + k));
                        zsum  += hv * static_cast<float>(whhGm.GetValue(rowIz + k));
                        nhsum += hv * static_cast<float>(whhGm.GetValue(rowIn + k));
                    }

                    rpre.SetValue(h_out, rsum);
                    zpre.SetValue(h_out, zsum);
                    nxpre.SetValue(h_out, nxsum);
                    nhpre.SetValue(h_out, nhsum);
                }

                // r,z = sigmoid
                AscendC::DataCopy(r, rpre, H);
                AscendC::DataCopy(z, zpre, H);
                SigmoidInplace(r, H);
                SigmoidInplace(z, H);

                // npre = nx + r*nh
                AscendC::Mul(npre, r, nhpre, H);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(npre, npre, nxpre, H);
                AscendC::PipeBarrier<PIPE_V>();

                // n = tanh(npre)
                TanhOut(n, npre, tmp1, H);

                // h_new = n + z*(hcur - n)
                AscendC::Sub(tmp1, hcur, n, H);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(tmp1, z, tmp1, H);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(tmp2, n, tmp1, H);
                AscendC::PipeBarrier<PIPE_V>();

                // write back hidden
                for (uint32_t k = 0; k < H; ++k) {
                    hstack.SetValue(hsOff + k, tmp2.GetValue(k));
                }

                // next layer input
                AscendC::DataCopy(xout, tmp2, H);
                AscendC::PipeBarrier<PIPE_V>();

                // swap xin/xout
                AscendC::LocalTensor<float> ttmp = xin;
                xin = xout;
                xout = ttmp;
            }

            // After last layer: xin holds last layer output for this t
            const uint64_t yBase = (static_cast<uint64_t>(t) * B + b) * H;
            for (uint32_t k = 0; k < H; ++k) {
                yGm.SetValue(yBase + k, static_cast<half>(xin.GetValue(k)));
            }
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t b = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (b < B) ProcessBatch(b);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::GlobalTensor<half> xGm, h0Gm, wihGm, whhGm, bihGm, bhhGm, yGm;
};

extern "C" __global__ __aicore__ void gru_custom(
    GM_ADDR x, GM_ADDR h0, GM_ADDR w_ih, GM_ADDR w_hh, GM_ADDR b_ih, GM_ADDR b_hh,
    GM_ADDR y, GM_ADDR /*workspace*/, GM_ADDR /*tiling*/)
{
    KernelGruCustom op;
    op.Init(x, h0, w_ih, w_hh, b_ih, b_hh, y);
    op.Process();
}
