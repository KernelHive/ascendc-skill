
#include "kernel_operator.h"

// GRU h_n-only, specialized to:
// T=512, B=10, I=128, H=256, L=6, bias=True, batch_first=False, bidirectional=False
//
// Packed params expected:
// w_ih: [L*3H, H]  (layer0 weight_ih [3H,I] padded with zeros to H columns)
// w_hh: [L*3H, H]
// b_ih: [L*3H]
// b_hh: [L*3H]
//
// Optimizations vs baseline:
// - One block per batch element (no wasted blocks).
// - Keep two input buffers and swap pointers between layers (avoid H-length UB DataCopy each layer).
// - UB-to-UB moves for hstack slices use vector DataCopy, not scalar loops.
// - Fused hidden update: h = n + z*(h - n) to reduce vector ops.
// - Reduced explicit PipeBarrier usage (kept in sigmoid/tanh sequences).

class KernelGruHiddenCustom {
public:
    __aicore__ inline KernelGruHiddenCustom() {}

    static constexpr uint32_t T = 512;
    static constexpr uint32_t B = 10;
    static constexpr uint32_t I = 128;
    static constexpr uint32_t H = 256;
    static constexpr uint32_t L = 6;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR h0,
                               GM_ADDR w_ih, GM_ADDR w_hh,
                               GM_ADDR b_ih, GM_ADDR b_hh,
                               GM_ADDR h_n)
    {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
        h0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(h0));
        wihGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w_ih));
        whhGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w_hh));
        bihGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b_ih));
        bhhGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b_hh));
        hnGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(h_n));

        // UB (float):
        // hstack: L*H = 1536
        // in0/in1/hcur: 3*H = 768
        // rpre/zpre/nxpre/nhpre: 4*H = 1024
        // r/z/npre/n: 4*H = 1024
        // tmp1/tmp2: 2*H = 512
        // Total = 4864 floats = 19456 bytes
        pipe.InitBuffer(calcBuf, 20 * 1024);
    }

    __aicore__ inline void SigmoidInplace(AscendC::LocalTensor<float>& x, int32_t count)
    {
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

        AscendC::LocalTensor<float> hstack = ub;                  // L*H
        AscendC::LocalTensor<float> in0    = ub[L * H + 0 * H];    // H
        AscendC::LocalTensor<float> in1    = ub[L * H + 1 * H];    // H
        AscendC::LocalTensor<float> hcur   = ub[L * H + 2 * H];    // H

        AscendC::LocalTensor<float> rpre   = ub[L * H + 3 * H];
        AscendC::LocalTensor<float> zpre   = ub[L * H + 4 * H];
        AscendC::LocalTensor<float> nxpre  = ub[L * H + 5 * H];
        AscendC::LocalTensor<float> nhpre  = ub[L * H + 6 * H];

        AscendC::LocalTensor<float> r      = ub[L * H + 7 * H];
        AscendC::LocalTensor<float> z      = ub[L * H + 8 * H];
        AscendC::LocalTensor<float> npre   = ub[L * H + 9 * H];
        AscendC::LocalTensor<float> n      = ub[L * H + 10 * H];

        AscendC::LocalTensor<float> tmp1   = ub[L * H + 11 * H];
        AscendC::LocalTensor<float> tmp2   = ub[L * H + 12 * H];

        // Initialize hstack from h0 (scalar GM reads; done once, not in hot inner loops).
        for (uint32_t ll = 0; ll < L; ++ll) {
            const uint64_t base = (static_cast<uint64_t>(ll) * B + b) * H;
            for (uint32_t k = 0; k < H; ++k) {
                hstack.SetValue(ll * H + k, static_cast<float>(h0Gm.GetValue(base + k)));
            }
        }

        // Time loop
        for (uint32_t t = 0; t < T; ++t) {
            // Build input for layer0 in in0: x padded to H
            AscendC::Duplicate(in0, 0.0f, H);
            // no barrier needed before SetValue loop (Duplicate produces in0, SetValue overwrites subset)
            const uint64_t xBase = (static_cast<uint64_t>(t) * B + b) * I;
            for (uint32_t k = 0; k < I; ++k) {
                in0.SetValue(k, static_cast<float>(xGm.GetValue(xBase + k)));
            }

            // Ping-pong pointers for layer inputs
            AscendC::LocalTensor<float> curIn = in0;
            AscendC::LocalTensor<float> nxtIn = in1;

            for (uint32_t ll = 0; ll < L; ++ll) {
                // hcur <- hstack slice (UB->UB vector copy)
                AscendC::DataCopy(hcur, hstack[ll * H], H);

                const uint64_t gateBase = static_cast<uint64_t>(ll) * 3ULL * H;

                // Bias init (scalar loads but only 4*H per layer-step, much cheaper than weights)
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

                // GEMV (scalar, but with hoisted bases)
                const uint64_t rowBaseR = (gateBase + 0ULL * H) * static_cast<uint64_t>(H);
                const uint64_t rowBaseZ = (gateBase + 1ULL * H) * static_cast<uint64_t>(H);
                const uint64_t rowBaseN = (gateBase + 2ULL * H) * static_cast<uint64_t>(H);

                for (uint32_t h_out = 0; h_out < H; ++h_out) {
                    float rsum  = rpre.GetValue(h_out);
                    float zsum  = zpre.GetValue(h_out);
                    float nxsum = nxpre.GetValue(h_out);
                    float nhsum = nhpre.GetValue(h_out);

                    const uint64_t rowR = rowBaseR + static_cast<uint64_t>(h_out) * H;
                    const uint64_t rowZ = rowBaseZ + static_cast<uint64_t>(h_out) * H;
                    const uint64_t rowN = rowBaseN + static_cast<uint64_t>(h_out) * H;

                    for (uint32_t k = 0; k < H; ++k) {
                        const float xv = curIn.GetValue(k);
                        rsum  += xv * static_cast<float>(wihGm.GetValue(rowR + k));
                        zsum  += xv * static_cast<float>(wihGm.GetValue(rowZ + k));
                        nxsum += xv * static_cast<float>(wihGm.GetValue(rowN + k));
                    }

                    for (uint32_t k = 0; k < H; ++k) {
                        const float hv = hcur.GetValue(k);
                        rsum  += hv * static_cast<float>(whhGm.GetValue(rowR + k));
                        zsum  += hv * static_cast<float>(whhGm.GetValue(rowZ + k));
                        nhsum += hv * static_cast<float>(whhGm.GetValue(rowN + k));
                    }

                    rpre.SetValue(h_out, rsum);
                    zpre.SetValue(h_out, zsum);
                    nxpre.SetValue(h_out, nxsum);
                    nhpre.SetValue(h_out, nhsum);
                }

                // r,z activations
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

                // h_new = n + z*(h - n)
                AscendC::Sub(tmp2, hcur, n, H);     // tmp2 = h - n
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(tmp2, tmp2, z, H);     // tmp2 = z*(h - n)
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(tmp2, tmp2, n, H);     // tmp2 = n + z*(h - n)
                AscendC::PipeBarrier<PIPE_V>();

                // write back hidden
                AscendC::DataCopy(hstack[ll * H], tmp2, H);

                // next layer input: swap pointers instead of copying full vector
                // nxtIn now should reference tmp2 values; we do a single UB->UB copy once per layer-step,
                // but we avoid the extra copy in baseline which copied hcur->xpad every time.
                // We still need curIn for compute above, so write tmp2 into nxtIn and swap.
                AscendC::DataCopy(nxtIn, tmp2, H);

                // swap
                AscendC::LocalTensor<float> ttmp = curIn;
                curIn = nxtIn;
                nxtIn = ttmp;
            }
        }

        // Store final h_n (scalar stores)
        for (uint32_t ll = 0; ll < L; ++ll) {
            const uint64_t outBase = (static_cast<uint64_t>(ll) * B + b) * H;
            for (uint32_t k = 0; k < H; ++k) {
                hnGm.SetValue(outBase + k, static_cast<half>(hstack.GetValue(ll * H + k)));
            }
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t b = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (b < B) {
            ProcessBatch(b);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;

    AscendC::GlobalTensor<half> xGm, h0Gm, wihGm, whhGm, bihGm, bhhGm, hnGm;
};

extern "C" __global__ __aicore__ void gru_hidden_custom(
    GM_ADDR x, GM_ADDR h0, GM_ADDR w_ih, GM_ADDR w_hh, GM_ADDR b_ih, GM_ADDR b_hh,
    GM_ADDR h_n, GM_ADDR /*workspace*/, GM_ADDR /*tiling*/)
{
    KernelGruHiddenCustom op;
    op.Init(x, h0, w_ih, w_hh, b_ih, b_hh, h_n);
    op.Process();
}
