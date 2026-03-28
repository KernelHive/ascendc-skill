/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file bev_pool.cpp
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
class KernelBevPool{
public:
    __aicore__ inline KernelBevPool() {}
    __aicore__ inline void Init(GM_ADDR depth, GM_ADDR feat, GM_ADDR ranks_depth, GM_ADDR ranks_feat, GM_ADDR ranks_bev, GM_ADDR interval_starts, GM_ADDR interval_lengths, GM_ADDR out,
        int32_t B, int32_t N, int32_t D, int32_t fH, int32_t fW, int32_t C, int32_t D_Z, int32_t D_Y, int32_t D_X, int32_t N_points, int32_t N_pillar) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->B = B;
        this->N = N;
        this->D = D;
        this->fH = fH;
        this->fW = fW;
        this->C = C;
        this->D_Z = D_Z;
        this->D_Y = D_Y;
        this->D_X = D_X;
        this->N_points = N_points;
        this->N_pillar = N_pillar;

        depthGm.SetGlobalBuffer((__gm__ DTYPE_DEPTH*)depth);
        featGm.SetGlobalBuffer((__gm__ DTYPE_DEPTH*)feat);
        ranks_depthGm.SetGlobalBuffer((__gm__ int32_t*)ranks_depth);
        ranks_featGm.SetGlobalBuffer((__gm__ int32_t*)ranks_feat);
        ranks_bevGm.SetGlobalBuffer((__gm__ int32_t*)ranks_bev);
        interval_startsGm.SetGlobalBuffer((__gm__ int32_t*)interval_starts);
        interval_lengthsGm.SetGlobalBuffer((__gm__ int32_t*)interval_lengths);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)out);

        InitGlobalMemory(outGm, this->B*this->C*this->D_Z*this->D_Y*this->D_X, (DTYPE_OUT)(0));

        pipe.InitBuffer(QueueTmp1, 32* sizeof(float));
        pipe.InitBuffer(QueueTmp2, 32 * sizeof(float));
        pipe.InitBuffer(QueueTmp3, 32 * sizeof(float));
    }
    __aicore__ inline void Process() {
        for(int32_t i=0;i<this->B*this->C*this->D_Z*this->D_Y*this->D_X;i++)
        {
            outGm.SetValue(i, 0);
        }
        int32_t b1 = this->D_Z * this->D_Y * this->D_X * this->C;
        int32_t b2 = this->D_Y * this->D_X * this->C;
        int32_t b3 = this->D_X * this->C;
        int32_t b4 = this->C;

        int32_t o1 = this->C*this->D_Z*this->D_Y*this->D_X;
        int32_t o2 = this->D_Z*this->D_Y*this->D_X;
        int32_t o3 = this->D_Y*this->D_X;
        int32_t o4 = this->D_X;

        int32_t start,length,end;
        int32_t rank_depth, rank_feat, rank_bev;
        int32_t mb, mdz, mdy, mdx, mc;
        int32_t remainder;
        float depth_val,feat_val,bev_feat;
        int32_t outaddr;
        for (int i = 0; i < this->N_pillar; i++) {
            start = interval_startsGm.GetValue(i);//interval_starts[i];
            length = interval_lengthsGm.GetValue(i);//interval_lengths[i];
            end = start + length;
    
            // 遍历区间内的每个点
            for (int j = start; j < end; j++) {
                rank_depth = ranks_depthGm.GetValue(j);//ranks_depth[j];
                rank_feat = ranks_featGm.GetValue(j);//ranks_feat[j];
                rank_bev = ranks_bevGm.GetValue(j);//ranks_bev[j];
              
                mb = rank_bev / b1;
                remainder = rank_bev - mb*b1;
                mdz = remainder / b2;
                remainder = remainder - mdz*b2;
                mdy = remainder / b3;
                remainder = remainder - mdy*b3;
                mdx = remainder / b4;
                mc = remainder - mdx*b4;

                outaddr = mb*o1 + mc*o2 + mdz*o3 + mdy*o4 + mdx;

                auto tmp1 = QueueTmp1.Get<DTYPE_DEPTH>();
                auto tmp2 = QueueTmp2.Get<DTYPE_DEPTH>();
                auto tmp3 = QueueTmp3.Get<DTYPE_DEPTH>();
                tmp1.SetValue(0, depthGm.GetValue(rank_depth));
                tmp2.SetValue(0, featGm.GetValue(rank_feat));
                tmp3.SetValue(0, outGm.GetValue(outaddr));
                Mul(tmp2, tmp1, tmp2, 1);
                Add(tmp1, tmp2, tmp3, 1);
                outGm.SetValue(outaddr, tmp1.GetValue(0));
            }
        }
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> QueueTmp1, QueueTmp2, QueueTmp3;
    
    GlobalTensor<DTYPE_DEPTH> depthGm;
    GlobalTensor<DTYPE_DEPTH> featGm;
    GlobalTensor<int32_t> ranks_depthGm;
    GlobalTensor<int32_t> ranks_featGm;
    GlobalTensor<int32_t> ranks_bevGm;
    GlobalTensor<int32_t> interval_startsGm;
    GlobalTensor<int32_t> interval_lengthsGm;
    GlobalTensor<DTYPE_OUT> outGm;

    int32_t B,N,D,fH,fW,C,D_Z,D_Y,D_X,N_points,N_pillar;
};

extern "C" __global__ __aicore__ void bev_pool(GM_ADDR depth, GM_ADDR feat, GM_ADDR ranks_depth, GM_ADDR ranks_feat, GM_ADDR ranks_bev, GM_ADDR interval_starts, GM_ADDR interval_lengths, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelBevPool op;
        op.Init(depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, out,
            tiling_data.B, tiling_data.N, tiling_data.D, tiling_data.fH, tiling_data.fW, tiling_data.C, tiling_data.D_Z, tiling_data.D_Y, tiling_data.D_X, tiling_data.N_points, tiling_data.N_pillar);
        op.Process();
}