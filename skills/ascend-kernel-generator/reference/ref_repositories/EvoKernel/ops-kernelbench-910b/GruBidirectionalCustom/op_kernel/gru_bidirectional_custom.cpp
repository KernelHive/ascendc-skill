
#include "kernel_operator.h"

// Specialized bidirectional stacked GRU returning output of last layer only.
// Fixed shapes: T=512, B=10, I=128, H=256, L=6, D=2, bias=True, batch_first=False.
//
// Host-packed params expected:
//   w_ih: [L*D*3H, 2H]  (layer0 padded from I to 2H; higher layers already [3H,2H])
//   w_hh: [L*D*3H, H]
//   b_ih: [L*D*3H]
//   b_hh: [L*D*3H]
// Row order: (l=0,d=0),(l=0,d=1),(l=1,d=0),(l=1,d=1),...
//
// Required GM scratch input (pre-allocated by Python):
//   ybuf: [L, T, B, 2H] float16
// Stores per-layer per-timestep concatenated outputs: [y_fwd, y_bwd].
//
// Output:
//   y: [T, B, 2H] float16   (output of last layer)

class KernelGruBidirectionalCustom {
public:
    __aicore__ inline KernelGruBidirectionalCustom() {}

    static constexpr uint32_t T = 512;
    static constexpr uint32_t B = 10;
    static constexpr uint32_t I = 128;
    static constexpr uint32_t H = 256;
    static constexpr uint32_t L = 6;
    static constexpr uint32_t D = 2;
    static constexpr uint32_t IN2H = 2 * H;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR h0,
                               GM_ADDR w_ih, GM_ADDR w_hh,
                               GM_ADDR b_ih, GM_ADDR b_hh,
                               GM_ADDR ybuf,
                               GM_ADDR y)
    {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
        h0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(h0));
        wihGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w_ih));
        whhGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w_hh));
        bihGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b_ih));
        bhhGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b_hh));
        ybufGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(ybuf));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));

        // UB arena (float): x2h(2H) + h(H) + 10*H temps = 14H floats = 3584 floats ~ 14KB.
        pipe.InitBuffer(calcBuf, 16384);
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

    __aicore__ inline void TanhOutViaSigmoid(AscendC::LocalTensor<float>& dst,
                                            const AscendC::LocalTensor<float>& src,
                                            AscendC::LocalTensor<float>& tmp,
                                            int32_t count)
    {
        // tanh(x) = 2*sigmoid(2x) - 1
        AscendC::Muls(tmp, src, 2.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        SigmoidInplace(tmp, count);
        AscendC::Muls(dst, tmp, 2.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(dst, dst, -1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GRUStep(uint32_t layer, uint32_t dir,
                                   const AscendC::LocalTensor<float>& x2h, // len 2H
                                   AscendC::LocalTensor<float>& h,         // len H (in/out)
                                   AscendC::LocalTensor<float>& rpre,
                                   AscendC::LocalTensor<float>& zpre,
                                   AscendC::LocalTensor<float>& nxpre,
                                   AscendC::LocalTensor<float>& nhpre,
                                   AscendC::LocalTensor<float>& r,
                                   AscendC::LocalTensor<float>& z,
                                   AscendC::LocalTensor<float>& npre,
                                   AscendC::LocalTensor<float>& n,
                                   AscendC::LocalTensor<float>& tmp1,
                                   AscendC::LocalTensor<float>& tmp2)
    {
        const uint64_t gateBase = (static_cast<uint64_t>(layer) * D + dir) * 3ULL * H;

        // Bias init: rpre = b_ir + b_hr; zpre = b_iz + b_hz; nxpre = b_in; nhpre = b_hn
        for (uint32_t ho = 0; ho < H; ++ho) {
            const uint64_t offR = gateBase + 0ULL * H + ho;
            const uint64_t offZ = gateBase + 1ULL * H + ho;
            const uint64_t offN = gateBase + 2ULL * H + ho;

            rpre.SetValue(ho, static_cast<float>(bihGm.GetValue(offR)) + static_cast<float>(bhhGm.GetValue(offR)));
            zpre.SetValue(ho, static_cast<float>(bihGm.GetValue(offZ)) + static_cast<float>(bhhGm.GetValue(offZ)));
            nxpre.SetValue(ho, static_cast<float>(bihGm.GetValue(offN)));
            nhpre.SetValue(ho, static_cast<float>(bhhGm.GetValue(offN)));
        }

        // GEMV accumulation (scalar): correctness-focused fixed-shape specialization.
        for (uint32_t ho = 0; ho < H; ++ho) {
            float rsum  = rpre.GetValue(ho);
            float zsum  = zpre.GetValue(ho);
            float nxsum = nxpre.GetValue(ho);
            float nhsum = nhpre.GetValue(ho);

            const uint64_t rowIr = (gateBase + 0ULL * H + ho) * static_cast<uint64_t>(IN2H);
            const uint64_t rowIz = (gateBase + 1ULL * H + ho) * static_cast<uint64_t>(IN2H);
            const uint64_t rowIn = (gateBase + 2ULL * H + ho) * static_cast<uint64_t>(IN2H);

            const uint64_t rowHr = (gateBase + 0ULL * H + ho) * static_cast<uint64_t>(H);
            const uint64_t rowHz = (gateBase + 1ULL * H + ho) * static_cast<uint64_t>(H);
            const uint64_t rowHn = (gateBase + 2ULL * H + ho) * static_cast<uint64_t>(H);

            for (uint32_t k = 0; k < IN2H; ++k) {
                const float xv = x2h.GetValue(k);
                rsum  += xv * static_cast<float>(wihGm.GetValue(rowIr + k));
                zsum  += xv * static_cast<float>(wihGm.GetValue(rowIz + k));
                nxsum += xv * static_cast<float>(wihGm.GetValue(rowIn + k));
            }
            for (uint32_t k = 0; k < H; ++k) {
                const float hv = h.GetValue(k);
                rsum  += hv * static_cast<float>(whhGm.GetValue(rowHr + k));
                zsum  += hv * static_cast<float>(whhGm.GetValue(rowHz + k));
                nhsum += hv * static_cast<float>(whhGm.GetValue(rowHn + k));
            }

            rpre.SetValue(ho, rsum);
            zpre.SetValue(ho, zsum);
            nxpre.SetValue(ho, nxsum);
            nhpre.SetValue(ho, nhsum);
        }

        // r,z = sigmoid(pre)
        AscendC::DataCopy(r, rpre, H);
        AscendC::DataCopy(z, zpre, H);
        SigmoidInplace(r, H);
        SigmoidInplace(z, H);

        // npre = nx + r * nh   (PyTorch: r multiplies only hidden contribution for n gate)
        AscendC::Mul(npre, r, nhpre, H);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(npre, npre, nxpre, H);
        AscendC::PipeBarrier<PIPE_V>();

        // n = tanh(npre)
        TanhOutViaSigmoid(n, npre, tmp1, H);

        // h_new = n + z*(h - n)
        AscendC::Sub(tmp1, h, n, H);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(tmp1, z, tmp1, H);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(tmp2, n, tmp1, H);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(h, tmp2, H);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessBatch(uint32_t b)
    {
        AscendC::LocalTensor<float> ub = calcBuf.Get<float>();

        AscendC::LocalTensor<float> x2h  = ub;              // 2H
        uint32_t off = IN2H;

        AscendC::LocalTensor<float> h    = ub[off];         off += H;

        AscendC::LocalTensor<float> rpre = ub[off];         off += H;
        AscendC::LocalTensor<float> zpre = ub[off];         off += H;
        AscendC::LocalTensor<float> nxpr = ub[off];         off += H;
        AscendC::LocalTensor<float> nhpr = ub[off];         off += H;

        AscendC::LocalTensor<float> r    = ub[off];         off += H;
        AscendC::LocalTensor<float> z    = ub[off];         off += H;
        AscendC::LocalTensor<float> npre = ub[off];         off += H;
        AscendC::LocalTensor<float> n    = ub[off];         off += H;

        AscendC::LocalTensor<float> tmp1 = ub[off];         off += H;
        AscendC::LocalTensor<float> tmp2 = ub[off];         off += H;

        // Process layers sequentially; ybuf provides exact stacked bidirectional semantics.
        for (uint32_t l = 0; l < L; ++l) {

            // ---- forward dir=0: write ybuf[l,t,b,0:H] ----
            {
                const uint64_t h0Base = (static_cast<uint64_t>(l * 2U + 0U) * B + b) * H;
                for (uint32_t k = 0; k < H; ++k) {
                    h.SetValue(k, static_cast<float>(h0Gm.GetValue(h0Base + k)));
                }
            }

            for (uint32_t t = 0; t < T; ++t) {
                if (l == 0) {
                    AscendC::Duplicate(x2h, 0.0f, IN2H);
                    AscendC::PipeBarrier<PIPE_V>();
                    const uint64_t xBase = (static_cast<uint64_t>(t) * B + b) * I;
                    for (uint32_t k = 0; k < I; ++k) {
                        x2h.SetValue(k, static_cast<float>(xGm.GetValue(xBase + k)));
                    }
                } else {
                    const uint64_t yBase = (((static_cast<uint64_t>(l - 1U) * T + t) * B + b) * IN2H);
                    for (uint32_t k = 0; k < IN2H; ++k) {
                        x2h.SetValue(k, static_cast<float>(ybufGm.GetValue(yBase + k)));
                    }
                }

                GRUStep(l, 0U, x2h, h, rpre, zpre, nxpr, nhpr, r, z, npre, n, tmp1, tmp2);

                const uint64_t yOutBase = (((static_cast<uint64_t>(l) * T + t) * B + b) * IN2H);
                for (uint32_t k = 0; k < H; ++k) {
                    ybufGm.SetValue(yOutBase + k, static_cast<half>(h.GetValue(k)));
                }
            }

            // ---- backward dir=1: write ybuf[l,t,b,H:2H] ----
            {
                const uint64_t h0Base = (static_cast<uint64_t>(l * 2U + 1U) * B + b) * H;
                for (uint32_t k = 0; k < H; ++k) {
                    h.SetValue(k, static_cast<float>(h0Gm.GetValue(h0Base + k)));
                }
            }

            for (int32_t tt = static_cast<int32_t>(T) - 1; tt >= 0; --tt) {
                const uint32_t t = static_cast<uint32_t>(tt);

                if (l == 0) {
                    AscendC::Duplicate(x2h, 0.0f, IN2H);
                    AscendC::PipeBarrier<PIPE_V>();
                    const uint64_t xBase = (static_cast<uint64_t>(t) * B + b) * I;
                    for (uint32_t k = 0; k < I; ++k) {
                        x2h.SetValue(k, static_cast<float>(xGm.GetValue(xBase + k)));
                    }
                } else {
                    const uint64_t yBase = (((static_cast<uint64_t>(l - 1U) * T + t) * B + b) * IN2H);
                    for (uint32_t k = 0; k < IN2H; ++k) {
                        x2h.SetValue(k, static_cast<float>(ybufGm.GetValue(yBase + k)));
                    }
                }

                GRUStep(l, 1U, x2h, h, rpre, zpre, nxpr, nhpr, r, z, npre, n, tmp1, tmp2);

                const uint64_t yOutBase = (((static_cast<uint64_t>(l) * T + t) * B + b) * IN2H) + H;
                for (uint32_t k = 0; k < H; ++k) {
                    ybufGm.SetValue(yOutBase + k, static_cast<half>(h.GetValue(k)));
                }
            }
        }

        // Copy last layer outputs into y: y[t,b,:] = ybuf[L-1,t,b,:]
        for (uint32_t t = 0; t < T; ++t) {
            const uint64_t srcBase = (((static_cast<uint64_t>(L - 1U) * T + t) * B + b) * IN2H);
            const uint64_t dstBase = ((static_cast<uint64_t>(t) * B + b) * IN2H);
            for (uint32_t k = 0; k < IN2H; ++k) {
                yGm.SetValue(dstBase + k, ybufGm.GetValue(srcBase + k));
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
    AscendC::GlobalTensor<half> xGm, h0Gm, wihGm, whhGm, bihGm, bhhGm, ybufGm, yGm;
};

extern "C" __global__ __aicore__ void gru_bidirectional_custom(
    GM_ADDR x, GM_ADDR h0, GM_ADDR w_ih, GM_ADDR w_hh, GM_ADDR b_ih, GM_ADDR b_hh,
    GM_ADDR ybuf,
    GM_ADDR y, GM_ADDR /*workspace*/, GM_ADDR /*tiling*/)
{
    KernelGruBidirectionalCustom op;
    op.Init(x, h0, w_ih, w_hh, b_ih, b_hh, ybuf, y);
    op.Process();
}
