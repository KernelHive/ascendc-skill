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
 * @file mmad.cpp
 */

#include "kernel_operator.h"

class KernelMmad {
public:
    __aicore__ inline KernelMmad()
    {
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
    }
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, uint32_t tileBBlockShape)
    {
        // set cube only
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
        cubeBlockShape = tileBBlockShape;
        CubeBlockSize = cubeBlockShape * cubeBlockShape;

        aGM.SetGlobalBuffer((__gm__ half *)a);
        bGM.SetGlobalBuffer((__gm__ half *)b);
        cGM.SetGlobalBuffer((__gm__ float *)c);
        biasGM.SetGlobalBuffer((__gm__ float *)bias);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
        pipe.InitBuffer(inQueueB2, 1, k * cubeBlockShape * sizeof(half));
        pipe.InitBuffer(outQueueCO1, 1, m * cubeBlockShape * sizeof(float));
        pipe.InitBuffer(inQueueC1, 1, n * sizeof(float));
        pipe.InitBuffer(inQueueC2, 1, cubeBlockShape * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();
        AscendC::LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        AscendC::LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
        AscendC::LocalTensor<float> bias1Local = inQueueC1.DeQue<float>();
        for(int i=0;i<n/cubeBlockShape;i++)
        {
            SplitB(b1Local,i);
            SplitBias(bias1Local,i);
            Compute(a2Local);
            CopyOut(i);
        }
        inQueueA2.FreeTensor(a2Local);
        inQueueB1.FreeTensor(b1Local);
        inQueueC1.FreeTensor(bias1Local);
    }

private:
    __aicore__ inline uint32_t CeilCubeBlock(uint32_t len) {
        return (len + cubeBlockShape - 1) / cubeBlockShape;
    }

    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        AscendC::LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
        AscendC::LocalTensor<float> bias1Local = inQueueC1.AllocTensor<float>();

        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = m;
        nd2nzA1Params.dValue = k;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = k;
        nd2nzA1Params.dstNzC0Stride = CeilCubeBlock(m) * cubeBlockShape;
        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, aGM, nd2nzA1Params);

        AscendC::Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = k;
        nd2nzB1Params.dValue = n;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = n;
        nd2nzB1Params.dstNzC0Stride = CeilCubeBlock(k) * cubeBlockShape;
        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1Local, bGM, nd2nzB1Params);

        AscendC::DataCopy(bias1Local, biasGM, n);
        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
        inQueueC1.EnQue(bias1Local);
    }

    __aicore__ inline void SplitA()
    {
        AscendC::LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
        AscendC::LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();

        uint32_t dstOffset = CeilCubeBlock(k) * CubeBlockSize;
        uint32_t srcOffset = CubeBlockSize;
 
        //nz to zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(k);
        loadDataParams.srcStride = CeilCubeBlock(m);
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        for (int i = 0; i < CeilCubeBlock(m); ++i) {
            AscendC::LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
        }

        inQueueA2.EnQue<half>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB(const AscendC::LocalTensor<half>& b1Local,const uint32_t bSplitIdx)
    {
        AscendC::LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

        // Nz -> Zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(k);
        loadDataParams.srcStride = 1;
        loadDataParams.ifTranspose = true;
        AscendC::LoadData(b2Local, b1Local[bSplitIdx * CeilCubeBlock(n) * CubeBlockSize], loadDataParams);

        inQueueB2.EnQue<half>(b2Local);
    }
    __aicore__ inline void SplitBias(const AscendC::LocalTensor<float>& bias1Local,const uint32_t bSplitIdx)
    {
        AscendC::LocalTensor<float> bias2Local = inQueueC2.AllocTensor<float>();
        AscendC::DataCopy(bias2Local, bias1Local[bSplitIdx*cubeBlockShape], cubeBlockShape);
        inQueueC2.EnQue<float>(bias2Local);
    }
    __aicore__ inline void Compute(const AscendC::LocalTensor<half> a2Local)
    {
        AscendC::LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        AscendC::LocalTensor<float> bias2Local = inQueueC2.DeQue<float>();
        AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = cubeBlockShape;
        mmadParams.k = k;
        mmadParams.cmatrixInitVal = false;
        AscendC::Mmad(c1Local, a2Local, b2Local, bias2Local, mmadParams);
        outQueueCO1.EnQue<float>(c1Local);
        inQueueB2.FreeTensor(b2Local);
        inQueueC2.FreeTensor(bias2Local);
    }
    __aicore__ inline void CopyOut(const uint32_t bSplitIdx )
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        // FixpipeParamsV220 : CO1 -> gm
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = cubeBlockShape;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = cubeBlockShape*sizeof(float); //表示源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.dstStride = n;

        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        // 默认设置 nz -> nd
        AscendC::Fixpipe(cGM[bSplitIdx*cubeBlockShape], c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
    // AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1_;  //分离架构无 CO2
    AscendC::TQue<AscendC::TPosition::C1, 1> inQueueC1;
    AscendC::TQue<AscendC::TPosition::C2, 1> inQueueC2;

    AscendC::GlobalTensor<half> aGM;
    AscendC::GlobalTensor<half> bGM;
    AscendC::GlobalTensor<float> cGM;
    AscendC::GlobalTensor<float> biasGM;
    uint16_t m = 32, k = 32, n = 32;
    uint16_t aSize, bSize, cSize;
    uint32_t cubeBlockShape,CubeBlockSize;
};


extern "C" __global__ __aicore__ void mmad(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {

    GET_TILING_DATA(tiling_data,tiling);
    KernelMmad op;
    op.Init(a,b,bias,c,tiling_data.tileBBlockShape);
    op.Process();
}