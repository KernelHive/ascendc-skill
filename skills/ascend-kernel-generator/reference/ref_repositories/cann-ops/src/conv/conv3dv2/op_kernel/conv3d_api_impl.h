/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_api_impl.h
 * \brief
 */

#ifndef CONV3D_API_IMPL_H
#define CONV3D_API_IMPL_H


#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "conv3d_util.h"
#include "conv3d_config.h"
#include "conv3d_common_func.h"
#include "conv_framework_util.h"
#include "conv_common_func.h"

#define QUE_DEPTH_SIZE_TWO 2

using namespace AscendC;

namespace conv3d {
template <typename Intf, class Config_, bool OutputOrder_ = false>
struct Conv3dApiImpl {
public:
    using Config = Config_;
    using ConvParam = typename Config_::ConvParam;
    constexpr static uint32_t ImplType = Config::implType;

    CONV_DEFINE_MEMBER(ConvParam, outputOrder, OutputOrder_, bool);
    CONV_DEFINE_MEMBER(ConvParam, l0pingpong, 0, int);
    CONV_DEFINE_MEMBER(ConvParam, bl1bypass, 0, int);
    CONV_DEFINE_MEMBER(ConvParam, groupConvType, 0, int);
    CONV_DEFINE_MEMBER(ConvParam, quantType, 0, int);
    CONV_DEFINE_MEMBER(ConvParam, loadChannelType, 0, int);

public:
    __aicore__ inline Conv3dApiImpl(){};

    CONV_REG_IMPL(Config, Conv3dFunc, Init);
    CONV_REG_IMPL(Config, ConvFunc, SetFmap);
    CONV_REG_IMPL(Config, ConvFunc, SetWeight);
    CONV_REG_IMPL(Config, ConvFunc, SetBias);
    CONV_REG_IMPL(Config, Conv3dFunc, SetOrgFmapShape);
    CONV_REG_IMPL(Config, Conv3dFunc, SetOrgWeightShape);
    CONV_REG_IMPL(Config, Conv3dFunc, SetOrgOutputShape);
    CONV_REG_IMPL(Config, Conv3dFunc, SetSingleFmapShape);
    CONV_REG_IMPL(Config, Conv3dFunc, SetSingleOutputShape);
    CONV_REG_IMPL(Config, Conv3dFunc, SetFmapStartPosition);
    CONV_REG_IMPL(Config, Conv3dFunc, BeginSetCrossFlag);
    CONV_REG_IMPL(Config, Conv3dFunc, EndWaitCrossFlag);
    CONV_REG_IMPL(Config, Conv3dFunc, SetWorkspace);
    CONV_REG_IMPL(Config, Conv3dFunc, SetVecScale);
    CONV_REG_IMPL(Config, Conv3dFunc, SetSubBlockIdx);
    CONV_REG_IMPL(Config, Conv3dFunc, LoadTotalCoreChannel);
    CONV_REG_IMPL(Config, Conv3dFunc, FreeTotalCoreChannel);
    CONV_REG_IMPL(Config, Conv3dFunc, SetGroupOptInfo);
    CONV_REG_IMPL(Config, Conv3dFunc, Iterate);
    CONV_REG_IMPL(Config, Conv3dFunc, IterateAll);
    CONV_REG_IMPL(Config, Conv3dFunc, GetTensorC);
    CONV_REG_IMPL(Config, Conv3dFunc, VecCompute);
    CONV_REG_IMPL(Config, ConvFunc, End);

    struct ContextData : public Config::ContextData {
        __aicore__ inline ContextData(){};

        const struct TConv3DTiling *__restrict conv3dTiling;

        TPipe pipe;

        using LoadAL1ToolsTmp1 = typename Conditional<Config::formatA == conv::ConvFormat::NCDHW,
            Conv3dFunc::LoadAL1WithPointWiseTools<Intf>, Conv3dFunc::LoadAL1Tools<Intf>>::type;
        using LoadAL1Tools =
            typename Conditional<outputOrder, Conv3dFunc::LoadAL1WithHwModeTools<Intf>, LoadAL1ToolsTmp1>::type;

        using LoadBL1ToolsTmp1 = typename Conditional<Config::formatA == conv::ConvFormat::NCDHW,
            Conv3dFunc::LoadBL1WithPointWiseTools<Intf>, Conv3dFunc::LoadBL1Tools<Intf>>::type;
        using LoadBL1ToolsTmp2 =
            typename Conditional<groupConvType, Conv3dFunc::LoadBL1WithGroupOptTools<Intf>, LoadBL1ToolsTmp1>::type;
        using LoadBL1Tools =
            typename Conditional<outputOrder, Conv3dFunc::LoadBL1WithHwModeTools<Intf>, LoadBL1ToolsTmp2>::type;

        using MMadTools = typename Conditional<Config::formatA == conv::ConvFormat::NCDHW,
            Conv3dFunc::MMadWithPointWiseTools<Intf>, Conv3dFunc::MMadTools<Intf>>::type;

        using LoadBL0ToolsTmp = typename Conditional<Config::formatA == conv::ConvFormat::NCDHW,
            Conv3dFunc::LoadBL0WithPointWiseTools<Intf>, Conv3dFunc::LoadBL0Tools<Intf>>::type;
        using LoadBL0Tools =
            typename Conditional<groupConvType, Conv3dFunc::LoadBL0WithGroupOptTools<Intf>, LoadBL0ToolsTmp>::type;

        using LoadAL0ToolsTmp1 = typename Conditional<Config::formatA == conv::ConvFormat::NCDHW,
            Conv3dFunc::LoadAL0WithPointWiseTools<Intf>, Conv3dFunc::LoadAL0Tools<Intf>>::type;
        using LoadAL0Tools =
            typename Conditional<outputOrder, Conv3dFunc::LoadAL0WithHwModeTools<Intf>, LoadAL0ToolsTmp1>::type;

        using LoadChannelWiseL1Tools = typename Conditional<
            Config::formatA == conv::ConvFormat::NCDHW,
            Conv3dFunc::LoadBiasL1WithPointWiseTools<Intf, typename Config::BiasT>,
            Conv3dFunc::LoadChannelWiseL1Tools<Intf, typename Config::BiasT>>::type;
        using LoadBiasL0Tools = typename Conditional<
            Config::formatA == conv::ConvFormat::NCDHW,
            Conv3dFunc::LoadBiasL0WithBroadcastTools<Intf>, Conv3dFunc::LoadBiasBtTools<Intf>>::type;

        using CopyOutToolsTmp1 = typename Conditional<Config::formatA == conv::ConvFormat::NCDHW,
            Conv3dFunc::CopyOutWithPointWiseTools<Intf>, Conv3dFunc::CopyOutTools<Intf>>::type;
        using CopyOutTools =
            typename Conditional<outputOrder, Conv3dFunc::CopyOutWithHwModeTools<Intf>, CopyOutToolsTmp1>::type;

        LoadAL1Tools loadAl1Ins;
        LoadBL1Tools loadBL1Ins;
        MMadTools madIns;
        LoadBL0Tools loadBL0Ins;
        LoadAL0Tools loadAL0Ins;
        LoadChannelWiseL1Tools loadBiasL1Ins;
        LoadBiasL0Tools loadBiasBTIns; // for pointwise situation, using l0a and l0b instead of bt.
        CopyOutTools copyOutIns;
        
        // GM Tensor
        GlobalTensor<typename Config::SrcAT> agm;
        GlobalTensor<typename Config::SrcBT> bgm;
        GlobalTensor<typename Config::BiasT> biasgm;
        GlobalTensor<typename Config::L0cT> workspacegm;
        GlobalTensor<typename Config::FP32T> scalegm;

        // LocalTensor
        LocalTensor<typename Config::SrcAT> al1;
        LocalTensor<typename Config::SrcBT> bl1;
        LocalTensor<typename Config::BiasT> biasL1;
        LocalTensor<typename Config::L0cT> biasBT;
        LocalTensor<typename Config::SrcAT> al0;
        LocalTensor<typename Config::SrcBT> bl0;
        LocalTensor<typename Config::SrcAT> al0Ping;
        LocalTensor<typename Config::SrcAT> al0Pong;
        LocalTensor<typename Config::SrcBT> bl0Ping;
        LocalTensor<typename Config::SrcBT> bl0Pong;
        LocalTensor<typename Config::L0cT> cl0;

        //vec queue
        TQue<QuePosition::VECIN, 1> queueUBin; 
        TQue<QuePosition::VECOUT, 1> queueUBout; 
        TQue<QuePosition::VECIN, 1> queueUBbias; 
        TQue<QuePosition::VECIN, 1> queueUBscale; 
        TBuf<TPosition::VECCALC> fp32BiasBuf;

        //vec tensor
        LocalTensor<typename Config::L0cT> ubin;
        LocalTensor<typename Config::BF16T> ubout;
        LocalTensor<typename Config::FP32T> ubbias;
        LocalTensor<typename Config::FP32T> ubscale;

        // Queue
        TQue<QuePosition::A1, QUE_DEPTH_SIZE_TWO> queueAL1;      // AL1
        TQue<QuePosition::B1, QUE_DEPTH_SIZE_TWO> queueBL1;      // BL1
        TQue<QuePosition::A1, 1> queueBiasL1;   // BiasL1
        TQue<TPosition::C2, 1> queueBiasBT;     // BT
        TQue<QuePosition::CO1, 1> queueCL0;  // CL0
        // Buffers
        using L0aBufType = typename Conditional<
            Config::formatA == conv::ConvFormat::NCDHW,
            TBuf<TPosition::B2>, TBuf<TPosition::A2>>::type;
        using L0bBufType = typename Conditional<
            Config::formatA == conv::ConvFormat::NCDHW,
            TBuf<TPosition::A2>, TBuf<TPosition::B2>>::type;

        L0aBufType l0aBuf;
        L0bBufType l0bBuf;
        LocalTensor<typename Config::L0cT> al0BiasB;
        LocalTensor<typename Config::L0cT> bl0BiasB;

        uint8_t enableBias = false;     // 是否有bias
        uint8_t isFirstIterate = true;  // 是否第一次Iterate
        uint8_t loadAL1Flag = true;     // 是否载入AL1的标志
        uint8_t loadBL1Flag = true;     // 是否载入BL1的标志
        uint8_t loadAL0Flag = true;     // 是否载入AL0的标志
        uint8_t loadBL0Flag = true;     // 是否载入BL0的标志
        uint8_t kAL1fullload = false;
        uint8_t kBL1fullload = false;
        uint8_t biasFullLoadFlag = false;
        uint8_t mL0IsDivisibleByWo = false; // mL0是否能整除wo的标志

        uint8_t freeAL1TensorFlag = false;
        uint8_t freeBL1TensorFlag = false;
        uint8_t isGroupOptDimTail = false;
        
        uint64_t kAL1Iter = 0;  // AL1上k方向迭代器
        uint64_t kBL1Iter = 0;  // BL1上k方向迭代器
        uint64_t mAL1Iter = 0;
        uint64_t nBL1Iter = 0;  // BL1上n方向迭代器
        uint64_t dOutIter = 0;
        uint64_t kIter = 0;     // k方向迭代器，从DDR到L0
        uint64_t kAL0Iter = 0;  // L1A 到L0方向的迭代器
        uint64_t kBL0Iter = 0;  // L1B 到L0方向的迭代器
        uint64_t mAL0Iter = 0;  // AL0上m方向迭代器
        uint64_t nBL0Iter = 0;  // BL0上n方向迭代器
        uint64_t groupOptIter = 0;  // groupopt方向迭代器

        uint64_t maxKAL1Iter = 0;
        uint64_t maxMAL1Iter = 0;
        uint64_t maxNBL1Iter = 0;
        uint64_t maxKBL1Iter = 0;
        uint64_t maxNL0Iter = 0;
        uint64_t maxML0Iter = 0;
        uint64_t maxKL0Iter = 0;
        uint64_t maxDOutIter = 0;
        uint64_t maxGroupOptIter = 0;

        uint64_t ddr2l1LoopN = 0;
        uint64_t l12l0LoopN = 0;
        uint64_t ddr2l1LoopD = 0;
        uint64_t ddr2l1LoopM = 0;
        uint64_t l12l0LoopM = 0;
        uint64_t ddr2l0LoopK = 0;

        // conv3d shape info
        uint64_t orgCi = 0;  //  fmap上cin大小
        uint64_t orgCo = 0;  //  weight上cout大小
        uint64_t orgDi = 0;
        uint64_t orgDo = 0;    //  weight上cout大小
        uint64_t orgHi = 0;    //  fmap上h大小
        uint64_t orgWi = 0;    //  fmap上w大小
        uint64_t orgHo = 0;    //  output上h大小
        uint64_t orgWo = 0;    //  output上w大小
        uint64_t kernelD = 0;  //  weight上d大小
        uint64_t kernelH = 0;  //  weight上h大小
        uint64_t kernelW = 0;  //  weight上w大小
        uint64_t strideD = 0;
        uint64_t strideH = 0;
        uint64_t strideW = 0;
        uint64_t dilationD = 0;
        uint64_t dilationH = 0;
        uint64_t dilationW = 0;
        uint64_t padHead = 0;
        uint64_t padTail = 0;
        uint64_t padUp = 0;
        uint64_t padDown = 0;
        uint64_t padLeft = 0;
        uint64_t padRight = 0;
        uint64_t singleCoreCin = 0;  // 单核上处理的Cin大小
        uint64_t singleCoreCo = 0;   // 单核上处理的Co大小
        uint64_t singleCoreM = 0;    // 单核上处理的M
        uint64_t singleCoreDo = 0;  // 单核上处理的Dout

        uint64_t dilatedKernelH = 0;
        uint64_t dilatedKernelW = 0;
        uint64_t dilatedKernelD = 0;
        uint64_t cin0 = 0;
        uint64_t cin1 = 0;
        uint64_t cout0 = 0;
        uint64_t cout1 = 0;

        uint64_t kernelHxkernelW = 0;
        uint64_t kernelHxkernelWxkernelD = 0;

        uint64_t kL0Tail = 0;
        uint64_t mAL1Tail = 0;
        uint64_t mAL0Tail = 0;
        uint64_t nL0Tail = 0;
        uint64_t nBL1Tail = 0;
        uint64_t multiKAL1 = 1;
        uint64_t multiKBL1 = 1;

        uint64_t mStartPos = 0;
        uint64_t diStartPos = 0;

        uint64_t orgCoAlignK0 = 0;
        uint64_t orgCoAlignN0 = 0;
        uint64_t nBL1TailAlign = 0;
        uint64_t sizeOfFmap = 0;
        uint64_t sizeOfWeight = 0;
        uint64_t sizeOfBias = 0;
        uint64_t sizeOfL0c = 0;
        uint64_t sizeOfDst = 0;

        // GroupOpt
        uint64_t groupKAL1 = 0;
        uint64_t groupKBL1 = 0;
        uint64_t groupKAL1Tail = 0;
        uint64_t groupKBL1Tail = 0;
        uint64_t singleCoreKL0 = 0;
        uint64_t preCorePerGroupSumCout = 0;
        uint64_t singleCoreGroupOpt = 0; // 单核上处理的GroutOpt
        uint64_t singleCoreCinTail = 0; // GroutOpt场景尾core的Cin
        uint64_t singleCoreCoutTail = 0; // GroutOpt场景尾core的Cout

        // HW_Mode
        uint64_t ddr2l1LoopHo = 0;
        uint64_t hoL1Iter = 0;
        uint64_t singleCoreHo = 0;
        uint64_t hiStartPos = 0;

        uint8_t preloadAL1DbFlag = false;
        uint8_t preloadABL1DbFlag = false;

        uint8_t workspaceDbFlag = 0;
        uint8_t C2VEvent = 0;
        uint8_t V2CEvent = 4;
        uint32_t subblockIdx = 0;
        uint16_t totalBlockCount = 0;
        uint16_t totalBlockLen = 0;
        uint32_t channelOffset = 0;
        uint32_t outNoffset = 0;
        uint32_t outMoffset = 0;
    };

    struct ImplDataType {
        __aicore__ inline ImplDataType(){};
    };
};

}  // namespace conv3d

#endif
