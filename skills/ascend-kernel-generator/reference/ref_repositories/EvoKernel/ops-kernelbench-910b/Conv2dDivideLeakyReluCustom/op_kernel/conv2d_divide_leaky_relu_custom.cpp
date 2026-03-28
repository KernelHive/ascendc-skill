
#include "kernel_operator.h"

// Fused conv3x3 + mul(inv_divisor) + leaky_relu, specialized:
// x: [128,8,128,128], w: [64,8,3,3], b: [64], y: [128,64,126,126].
//
// This round: reduce scalar-bound behavior by increasing ILP and reducing loop/control overhead.
// - Keep (n,co,oh_pair) tasking but compute width tiles of 8 outputs at once (8 accumulators).
// - Still compute two rows (oh and oh+1) per task.
// - Hoist repeated address arithmetic and unroll cin=8.
// - Branchless leaky-relu.

class KernelConv2dDivideLeakyReluCustom {
public:
    __aicore__ inline KernelConv2dDivideLeakyReluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kh, uint32_t kw,
                               uint32_t hout, uint32_t wout,
                               uint32_t stride, uint32_t pad, uint32_t dilation, uint32_t groups,
                               float inv_divisor, float leaky_slope,
                               uint32_t total_pair_tasks, uint32_t pairs_per_block,
                               uint32_t hw_out, uint32_t cout_hw_out, uint32_t cin_hw_in, uint32_t hw_in)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout; this->kh = kh; this->kw = kw;
        this->hout = hout; this->wout = wout;
        this->stride = stride; this->pad = pad; this->dilation = dilation; this->groups = groups;
        this->inv_divisor = inv_divisor;
        this->leaky_slope = leaky_slope;
        this->total_pair_tasks = total_pair_tasks;
        this->pairs_per_block = pairs_per_block;

        this->hw_out = hw_out;
        this->cout_hw_out = cout_hw_out;
        this->cin_hw_in = cin_hw_in;
        this->hw_in = hw_in;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cout) * cin * kh * kw;
        const uint64_t bSize = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n) * cout * hout * wout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);
    }

    __aicore__ inline float LeakyReluBranchless(float v) const
    {
        const float pos = (v > 0.0f) ? v : 0.0f;
        const float neg = (v < 0.0f) ? v : 0.0f;
        return pos + neg * leaky_slope;
    }

    __aicore__ inline void ComputeTwoRowsTiled8(uint32_t ni, uint32_t co, uint32_t oh0)
    {
        const uint32_t oh1 = oh0 + 1u;
        const bool doRow1 = (oh1 < hout);

        const uint64_t win64 = static_cast<uint64_t>(win);
        const uint64_t hw_in64 = static_cast<uint64_t>(hw_in);
        const uint64_t cin_hw_in64 = static_cast<uint64_t>(cin_hw_in);

        const uint64_t xNBase = static_cast<uint64_t>(ni) * cin_hw_in64;
        const uint64_t wBase = static_cast<uint64_t>(co) * 72ull;
        const float bias = bGm.GetValue(static_cast<uint64_t>(co));

        const uint64_t yBase0 = (static_cast<uint64_t>(ni) * static_cast<uint64_t>(cout_hw_out)) +
                                (static_cast<uint64_t>(co) * static_cast<uint64_t>(hw_out)) +
                                (static_cast<uint64_t>(oh0) * static_cast<uint64_t>(wout));
        const uint64_t yBase1 = yBase0 + static_cast<uint64_t>(wout);

        // Process ow in tiles of 8 to increase ILP (8 accumulators).
        for (uint32_t ow0 = 0; ow0 < wout; ow0 += 8u) {
            const uint32_t remain = wout - ow0;
            const uint32_t tile = (remain >= 8u) ? 8u : remain;

            float acc0[8];
            float acc1[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) { acc0[i] = bias; acc1[i] = bias; }

            // Base within channel for row0 at (oh0, ow0)
            const uint64_t xOHW0 = static_cast<uint64_t>(oh0) * win64 + static_cast<uint64_t>(ow0);

            #pragma unroll
            for (uint32_t ci = 0; ci < 8; ++ci) {
                const uint64_t xCBase0 = xNBase + static_cast<uint64_t>(ci) * hw_in64 + xOHW0;
                const uint64_t wCBase  = wBase  + static_cast<uint64_t>(ci) * 9ull;

                const float w00 = wGm.GetValue(wCBase + 0);
                const float w01 = wGm.GetValue(wCBase + 1);
                const float w02 = wGm.GetValue(wCBase + 2);
                const float w10 = wGm.GetValue(wCBase + 3);
                const float w11 = wGm.GetValue(wCBase + 4);
                const float w12 = wGm.GetValue(wCBase + 5);
                const float w20 = wGm.GetValue(wCBase + 6);
                const float w21 = wGm.GetValue(wCBase + 7);
                const float w22 = wGm.GetValue(wCBase + 8);

                // Row pointers for row0 patch
                const uint64_t r0 = xCBase0;
                const uint64_t r1 = r0 + win64;
                const uint64_t r2 = r1 + win64;

                // For row1 patch, just +win
                const uint64_t s0 = r0 + win64;
                const uint64_t s1 = s0 + win64;
                const uint64_t s2 = s1 + win64;

                // Unrolled ow tile (use bounds only for tail tile)
                #pragma unroll
                for (uint32_t i = 0; i < 8; ++i) {
                    if (i >= tile) break;

                    // row0: 3x3 taps at ow0+i
                    const uint64_t o = static_cast<uint64_t>(i);
                    const float x00 = xGm.GetValue(r0 + o + 0);
                    const float x01 = xGm.GetValue(r0 + o + 1);
                    const float x02 = xGm.GetValue(r0 + o + 2);
                    const float x10 = xGm.GetValue(r1 + o + 0);
                    const float x11 = xGm.GetValue(r1 + o + 1);
                    const float x12 = xGm.GetValue(r1 + o + 2);
                    const float x20 = xGm.GetValue(r2 + o + 0);
                    const float x21 = xGm.GetValue(r2 + o + 1);
                    const float x22 = xGm.GetValue(r2 + o + 2);

                    acc0[i] += x00 * w00 + x01 * w01 + x02 * w02;
                    acc0[i] += x10 * w10 + x11 * w11 + x12 * w12;
                    acc0[i] += x20 * w20 + x21 * w21 + x22 * w22;

                    if (doRow1) {
                        const float y00 = xGm.GetValue(s0 + o + 0);
                        const float y01 = xGm.GetValue(s0 + o + 1);
                        const float y02 = xGm.GetValue(s0 + o + 2);
                        const float y10 = xGm.GetValue(s1 + o + 0);
                        const float y11 = xGm.GetValue(s1 + o + 1);
                        const float y12 = xGm.GetValue(s1 + o + 2);
                        const float y20 = xGm.GetValue(s2 + o + 0);
                        const float y21 = xGm.GetValue(s2 + o + 1);
                        const float y22 = xGm.GetValue(s2 + o + 2);

                        acc1[i] += y00 * w00 + y01 * w01 + y02 * w02;
                        acc1[i] += y10 * w10 + y11 * w11 + y12 * w12;
                        acc1[i] += y20 * w20 + y21 * w21 + y22 * w22;
                    }
                }
            }

            // Epilogue + store
            #pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                if (i >= tile) break;

                float v0 = acc0[i] * inv_divisor;
                v0 = LeakyReluBranchless(v0);
                yGm.SetValue(yBase0 + static_cast<uint64_t>(ow0 + i), v0);

                if (doRow1) {
                    float v1 = acc1[i] * inv_divisor;
                    v1 = LeakyReluBranchless(v1);
                    yGm.SetValue(yBase1 + static_cast<uint64_t>(ow0 + i), v1);
                }
            }
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t startTask = bid * pairs_per_block;
        uint32_t endTask = startTask + pairs_per_block;
        if (endTask > total_pair_tasks) endTask = total_pair_tasks;
        if (startTask >= endTask) return;

        const uint32_t hout_pairs = (hout + 1u) / 2u;
        const uint32_t cout_hpairs = cout * hout_pairs;

        for (uint32_t t = startTask; t < endTask; ++t) {
            const uint32_t ni = t / cout_hpairs;
            const uint32_t rem0 = t - ni * cout_hpairs;
            const uint32_t co = rem0 / hout_pairs;
            const uint32_t ohp = rem0 - co * hout_pairs;
            const uint32_t oh0 = ohp * 2u;

            ComputeTwoRowsTiled8(ni, co, oh0);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n{}, cin{}, hin{}, win{};
    uint32_t cout{}, kh{}, kw{};
    uint32_t hout{}, wout{};
    uint32_t stride{}, pad{}, dilation{}, groups{};
    float inv_divisor{};
    float leaky_slope{};

    uint32_t total_pair_tasks{}, pairs_per_block{};
    uint32_t hw_out{}, cout_hw_out{}, cin_hw_in{}, hw_in{};
};

extern "C" __global__ __aicore__ void conv2d_divide_leaky_relu_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv2dDivideLeakyReluCustom op;
    op.Init(x, weight, bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.hout, t.wout,
            t.stride, t.pad, t.dilation, t.groups,
            t.inv_divisor, t.leaky_slope,
            t.total_pair_tasks, t.pairs_per_block,
            t.hw_out, t.cout_hw_out, t.cin_hw_in, t.hw_in);
    op.Process();
}
