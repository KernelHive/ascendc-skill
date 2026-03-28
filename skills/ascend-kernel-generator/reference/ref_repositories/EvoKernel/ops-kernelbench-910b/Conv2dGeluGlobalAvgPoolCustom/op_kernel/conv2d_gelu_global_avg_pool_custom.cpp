
#include "kernel_operator.h"

// Correctness-first fused op:
//   y[n,co] = mean_{oh,ow} GELU( Conv2d(x,w,b)[n,co,oh,ow] )
//
// Specialized:
//   x: [128,8,256,256], w:[64,8,3,3], b:[64]
//   stride=1 pad=0 dil=1 groups=1
//   conv out: [128,64,254,254], y: [128,64]

class KernelConv2dGeluGlobalAvgPoolCustom {
public:
    __aicore__ inline KernelConv2dGeluGlobalAvgPoolCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kh, uint32_t kw,
                               uint32_t stride, uint32_t pad, uint32_t dilation, uint32_t groups,
                               uint32_t hout, uint32_t wout,
                               float inv_hwout,
                               uint32_t total_tasks, uint32_t tasks_per_block,
                               AscendC::TPipe* pipe)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout; this->kh = kh; this->kw = kw;
        this->stride = stride; this->pad = pad; this->dilation = dilation; this->groups = groups;
        this->hout = hout; this->wout = wout;
        this->inv_hwout = inv_hwout;
        this->total_tasks = total_tasks;
        this->tasks_per_block = tasks_per_block;
        this->pipe = pipe;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cout) * (cin / groups) * kh * kw;
        const uint64_t bSize = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n) * cout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        pipe->InitBuffer(qTile, 1, TILE_ELEMS * sizeof(float));
        pipe->InitBuffer(tmpBuf, TMP_BYTES);
    }

    __aicore__ inline float ConvAt(uint32_t ni, uint32_t co, uint32_t oh, uint32_t ow) const
    {
        // Specialized: stride=1 pad=0 dilation=1 groups=1 kh=kw=3.
        float acc = bGm.GetValue(static_cast<uint64_t>(co));

        const uint64_t cin64 = static_cast<uint64_t>(cin);
        const uint64_t hin64 = static_cast<uint64_t>(hin);
        const uint64_t win64 = static_cast<uint64_t>(win);

        // Input top-left = (oh,ow)
        for (uint32_t ci = 0; ci < cin; ++ci) {
            const uint64_t xBase =
                ((static_cast<uint64_t>(ni) * cin64 + static_cast<uint64_t>(ci)) * hin64 +
                 static_cast<uint64_t>(oh)) * win64 + static_cast<uint64_t>(ow);

            const uint64_t wBase =
                ((static_cast<uint64_t>(co) * cin64 + static_cast<uint64_t>(ci)) * 3ull) * 3ull;

            acc += xGm.GetValue(xBase + 0) * wGm.GetValue(wBase + 0);
            acc += xGm.GetValue(xBase + 1) * wGm.GetValue(wBase + 1);
            acc += xGm.GetValue(xBase + 2) * wGm.GetValue(wBase + 2);

            const uint64_t xBase1 = xBase + win64;
            const uint64_t wBase1 = wBase + 3;
            acc += xGm.GetValue(xBase1 + 0) * wGm.GetValue(wBase1 + 0);
            acc += xGm.GetValue(xBase1 + 1) * wGm.GetValue(wBase1 + 1);
            acc += xGm.GetValue(xBase1 + 2) * wGm.GetValue(wBase1 + 2);

            const uint64_t xBase2 = xBase + 2 * win64;
            const uint64_t wBase2 = wBase + 6;
            acc += xGm.GetValue(xBase2 + 0) * wGm.GetValue(wBase2 + 0);
            acc += xGm.GetValue(xBase2 + 1) * wGm.GetValue(wBase2 + 1);
            acc += xGm.GetValue(xBase2 + 2) * wGm.GetValue(wBase2 + 2);
        }
        return acc;
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t start = bid * tasks_per_block;
        uint32_t end = start + tasks_per_block;
        if (end > total_tasks) end = total_tasks;
        if (start >= end) return;

        const uint32_t hwOut = hout * wout;

        AscendC::LocalTensor<float> tile = qTile.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> tmp = tmpBuf.AllocTensor<uint8_t>();

        for (uint32_t task = start; task < end; ++task) {
            const uint32_t ni = task / cout;
            const uint32_t co = task - ni * cout;

            float sum = 0.0f;
            uint32_t p = 0;
            while (p < hwOut) {
                uint32_t cnt = TILE_ELEMS;
                const uint32_t remain = hwOut - p;
                if (remain < cnt) cnt = remain;

                // Fill tile[0:cnt) with conv+bias
                for (uint32_t i = 0; i < cnt; ++i) {
                    const uint32_t lin = p + i;
                    const uint32_t oh = lin / wout;
                    const uint32_t ow = lin - oh * wout;
                    tile.SetValue(i, ConvAt(ni, co, oh, ow));
                }
                for (uint32_t i = cnt; i < TILE_ELEMS; ++i) tile.SetValue(i, 0.0f);

                // GELU in place on full TILE to satisfy vector API; extra padded values are zero.
                AscendC::FasterGeluV2<float, false, true>(tile, tile, tmp, TILE_ELEMS);

                // Accumulate only valid cnt
                for (uint32_t i = 0; i < cnt; ++i) sum += tile.GetValue(i);

                p += cnt;
            }

            yGm.SetValue(static_cast<uint64_t>(ni) * static_cast<uint64_t>(cout) + static_cast<uint64_t>(co),
                         sum * inv_hwout);
        }

        tmpBuf.FreeTensor(tmp);
        qTile.FreeTensor(tile);
    }

private:
    static constexpr uint32_t TILE_ELEMS = 256;
    static constexpr uint32_t TMP_BYTES = 8192;

    AscendC::TPipe* pipe{};
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qTile;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n{}, cin{}, hin{}, win{};
    uint32_t cout{}, kh{}, kw{};
    uint32_t stride{}, pad{}, dilation{}, groups{};
    uint32_t hout{}, wout{};
    float inv_hwout{};
    uint32_t total_tasks{}, tasks_per_block{};
};

extern "C" __global__ __aicore__ void conv2d_gelu_global_avg_pool_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    AscendC::TPipe pipe;
    KernelConv2dGeluGlobalAvgPoolCustom op;
    op.Init(x, weight, bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.stride, t.pad, t.dilation, t.groups,
            t.hout, t.wout,
            t.inv_hwout,
            t.total_tasks, t.tasks_per_block,
            &pipe);
    op.Process();
}
