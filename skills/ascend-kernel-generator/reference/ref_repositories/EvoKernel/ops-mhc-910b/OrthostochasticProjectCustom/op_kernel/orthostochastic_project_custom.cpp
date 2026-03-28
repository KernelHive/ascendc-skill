
#include "kernel_operator.h"

using F32 = float;
using I32 = int32_t;

static constexpr uint32_t MAX_N = 32;
static constexpr uint32_t MAX_ELEMS = MAX_N * MAX_N;

class KernelOrthostochasticProjectCustom {
public:
    __aicore__ inline KernelOrthostochasticProjectCustom() {}

    __aicore__ inline void Init(GM_ADDR logits,
                               GM_ADDR steps, GM_ADDR eps, GM_ADDR a, GM_ADDR b, GM_ADDR c,
                               GM_ADDR out,
                               uint32_t n0, uint32_t n1, uint32_t m, uint32_t n,
                               uint32_t mn, uint32_t transpose)
    {
        this->n0 = n0; this->n1 = n1;
        this->m = m; this->n = n;
        this->mn = mn;
        this->transpose = transpose;

        logitsGm.SetGlobalBuffer((__gm__ F32*)logits, (uint64_t)n0 * (uint64_t)n1);
        outGm.SetGlobalBuffer((__gm__ F32*)out, (uint64_t)n0 * (uint64_t)n1);

        stepsGm.SetGlobalBuffer((__gm__ I32*)steps, (uint64_t)1);
        epsGm.SetGlobalBuffer((__gm__ F32*)eps, (uint64_t)1);
        aGm.SetGlobalBuffer((__gm__ F32*)a, (uint64_t)1);
        bGm.SetGlobalBuffer((__gm__ F32*)b, (uint64_t)1);
        cGm.SetGlobalBuffer((__gm__ F32*)c, (uint64_t)1);

        // Minimal UB: input/output only (we keep math in registers for 8x8).
        // Allocate max 32x32 to stay safe for fallback path; still only 2 buffers.
        pipe.InitBuffer(xUb,   (uint32_t)(MAX_ELEMS * sizeof(F32)));
        pipe.InitBuffer(outUb, (uint32_t)(MAX_ELEMS * sizeof(F32)));
    }

    __aicore__ inline void Process()
    {
        if (AscendC::GetBlockIdx() != 0) return;
        if (n0 == 0 || n1 == 0) return;

        I32 steps = stepsGm.GetValue(0);
        if (steps < 0) steps = 0;
        if (steps > 50) steps = 50;

        F32 eps = epsGm.GetValue(0);
        if (!(eps > (F32)0.0f)) eps = (F32)1e-7f;

        const F32 a = aGm.GetValue(0);
        const F32 b = bGm.GetValue(0);
        const F32 c = cGm.GetValue(0);

        AscendC::LocalTensor<F32> Xbuf = xUb.Get<F32>();
        AscendC::LocalTensor<F32> Obuf = outUb.Get<F32>();

        // Fast path: 8x8 (the workload's common case), no transpose needed.
        if (n0 == 8u && n1 == 8u) {
            AscendC::DataCopy(Xbuf, logitsGm, 64u);

            // Load into registers
            F32 X[8][8];
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j) {
                    X[i][j] = Xbuf.GetValue(i * 8u + j);
                }
            }

            // Frobenius norm
            F32 sumsq = (F32)0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j) {
                    F32 v = X[i][j];
                    sumsq += v * v;
                }
            }
            F32 norm = SqrtNewton(sumsq) + eps;
            if (norm < (F32)1e-12f) norm = (F32)1e-12f;
            F32 invNorm = (F32)1.0f / norm;

            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j) {
                    X[i][j] *= invNorm;
                }
            }

            // Newton–Schulz iterations in registers
            for (I32 it = 0; it < steps; ++it) {
                F32 A[8][8];
                F32 AA[8][8];

                // A = X * X^T
                #pragma unroll
                for (uint32_t i = 0; i < 8u; ++i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 8u; ++j) {
                        F32 acc = (F32)0.0f;
                        #pragma unroll
                        for (uint32_t p = 0; p < 8u; ++p) acc += X[i][p] * X[j][p];
                        A[i][j] = acc;
                    }
                }

                // AA = A * A
                #pragma unroll
                for (uint32_t i = 0; i < 8u; ++i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 8u; ++j) {
                        F32 acc = (F32)0.0f;
                        #pragma unroll
                        for (uint32_t p = 0; p < 8u; ++p) acc += A[i][p] * A[p][j];
                        AA[i][j] = acc;
                    }
                }

                // Xnext = a*X + (bA + cAA) * X
                F32 XN[8][8];
                #pragma unroll
                for (uint32_t i = 0; i < 8u; ++i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 8u; ++j) {
                        F32 acc = (F32)0.0f;
                        #pragma unroll
                        for (uint32_t p = 0; p < 8u; ++p) {
                            F32 bm = b * A[i][p] + c * AA[i][p];
                            acc += bm * X[p][j];
                        }
                        XN[i][j] = a * X[i][j] + acc;
                    }
                }

                #pragma unroll
                for (uint32_t i = 0; i < 8u; ++i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < 8u; ++j) {
                        X[i][j] = XN[i][j];
                    }
                }
            }

            // Square and store to UB contiguously
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j) {
                    F32 v = X[i][j];
                    Obuf.SetValue(i * 8u + j, v * v);
                }
            }
            AscendC::DataCopy(outGm, Obuf, 64u);
            return;
        }

        // Generic fallback for <=32x32, kept correct but not heavily optimized.
        // Still reduces some overhead by using contiguous GM<->UB copies and no full-buffer clears.
        AscendC::DataCopy(Xbuf, logitsGm, (uint32_t)(n0 * n1));

        // Build X (m x n) in-place inside Xbuf in row-major at start of buffer.
        // If transpose, we do a simple transpose into Obuf then swap buffers by copy (mn only).
        if (transpose != 0u) {
            for (uint32_t i = 0; i < m; ++i) {
                for (uint32_t j = 0; j < n; ++j) {
                    Obuf.SetValue(i * n + j, Xbuf.GetValue(j * n1 + i));
                }
            }
            for (uint32_t k = 0; k < mn; ++k) {
                Xbuf.SetValue(k, Obuf.GetValue(k));
            }
        } else {
            // already correct layout at top-left for m=n0,n=n1
        }

        // Normalize
        F32 sumsq = (F32)0.0f;
        for (uint32_t k = 0; k < mn; ++k) {
            F32 v = Xbuf.GetValue(k);
            sumsq += v * v;
        }
        F32 norm = SqrtNewton(sumsq) + eps;
        if (norm < (F32)1e-12f) norm = (F32)1e-12f;
        F32 invNorm = (F32)1.0f / norm;
        for (uint32_t k = 0; k < mn; ++k) {
            Xbuf.SetValue(k, Xbuf.GetValue(k) * invNorm);
        }

        // Iterations (scalar fallback)
        for (I32 it = 0; it < steps; ++it) {
            // Use Obuf as scratch for A (m*m) stored at start, and XN stored after A.
            // Layout: [A(mm)] [XN(mn)] within Obuf; safe since both <= 1024.
            const uint32_t mm = m * m;
            for (uint32_t i = 0; i < m; ++i) {
                for (uint32_t j = 0; j < m; ++j) {
                    F32 acc = (F32)0.0f;
                    for (uint32_t p = 0; p < n; ++p) {
                        acc += Xbuf.GetValue(i * n + p) * Xbuf.GetValue(j * n + p);
                    }
                    Obuf.SetValue(i * m + j, acc);
                }
            }

            // AA on the fly and compute XN
            for (uint32_t i = 0; i < m; ++i) {
                for (uint32_t j = 0; j < n; ++j) {
                    F32 acc = (F32)0.0f;
                    for (uint32_t p = 0; p < m; ++p) {
                        // compute AA(i,p) = sum_q A(i,q)*A(q,p)
                        F32 aa = (F32)0.0f;
                        for (uint32_t q = 0; q < m; ++q) {
                            aa += Obuf.GetValue(i * m + q) * Obuf.GetValue(q * m + p);
                        }
                        F32 bm = b * Obuf.GetValue(i * m + p) + c * aa;
                        acc += bm * Xbuf.GetValue(p * n + j);
                    }
                    Obuf.SetValue(mm + i * n + j, a * Xbuf.GetValue(i * n + j) + acc);
                }
            }
            for (uint32_t k = 0; k < mn; ++k) {
                Xbuf.SetValue(k, Obuf.GetValue((m * m) + k));
            }
        }

        // Write squared output with transpose-back if needed (use Obuf as staging of n0*n1)
        if (transpose == 0u) {
            for (uint32_t i = 0; i < n0; ++i) {
                for (uint32_t j = 0; j < n1; ++j) {
                    F32 v = Xbuf.GetValue(i * n1 + j);
                    Obuf.SetValue(i * n1 + j, v * v);
                }
            }
        } else {
            for (uint32_t i = 0; i < n0; ++i) {
                for (uint32_t j = 0; j < n1; ++j) {
                    F32 v = Xbuf.GetValue(j * n0 + i);
                    Obuf.SetValue(i * n1 + j, v * v);
                }
            }
        }
        AscendC::DataCopy(outGm, Obuf, (uint32_t)(n0 * n1));
    }

private:
    __aicore__ inline F32 SqrtNewton(F32 x)
    {
        if (x <= (F32)0.0f) return (F32)0.0f;
        // 4 Newton steps
        F32 y = x;
        y = (y + x / y) * (F32)0.5f;
        y = (y + x / y) * (F32)0.5f;
        y = (y + x / y) * (F32)0.5f;
        y = (y + x / y) * (F32)0.5f;
        return y;
    }

private:
    AscendC::TPipe pipe;

    AscendC::GlobalTensor<F32> logitsGm, outGm;
    AscendC::GlobalTensor<I32> stepsGm;
    AscendC::GlobalTensor<F32> epsGm, aGm, bGm, cGm;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> xUb, outUb;

    uint32_t n0{}, n1{}, m{}, n{}, mn{}, transpose{};
};

extern "C" __global__ __aicore__ void orthostochastic_project_custom(
    GM_ADDR logits,
    GM_ADDR steps, GM_ADDR eps, GM_ADDR a, GM_ADDR b, GM_ADDR c,
    GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelOrthostochasticProjectCustom op;
    op.Init(logits, steps, eps, a, b, c, out,
            tiling_data.n0, tiling_data.n1,
            tiling_data.m, tiling_data.n,
            tiling_data.mn, tiling_data.transpose);
    op.Process();
}
