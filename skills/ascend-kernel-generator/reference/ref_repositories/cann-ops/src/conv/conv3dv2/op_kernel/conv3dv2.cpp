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
 * \file conv3dv2.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "conv3dv2_with_groups.h"
#include "conv3dv2_pointwise.h"
#include "conv3dv2_hw_mode.h"
#include "conv3dv2.h"

using namespace AscendC;

#ifndef DTYPE_BIAS
#define DTYPE_BIAS float
#endif

#if defined(FORMAT_X) && FORMAT_X == FORMAT_NCDHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_NCDHW && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCDHW
constexpr ConvFormat aFormat = ConvFormat::NCDHW;
constexpr ConvFormat bFormat = ConvFormat::NCDHW;
constexpr ConvFormat cFormat = ConvFormat::NCDHW;
constexpr ConvBL1ByPass bL1ByPassFlag = ConvBL1ByPass::BYPASS_OFF;
#else
constexpr ConvFormat aFormat = ConvFormat::NDC1HWC0;
constexpr ConvFormat bFormat = ConvFormat::FRACTAL_Z_3D;
constexpr ConvFormat cFormat = ConvFormat::NDC1HWC0;
constexpr ConvBL1ByPass bL1ByPassFlag = ConvBL1ByPass::BYPASS_ON;
#endif

#define CONV3DV2_PROCESS_BASE(aConvType,                                                                       \
    bConvType,                                                                                                 \
    cConvType,                                                                                                 \
    biasConvType,                                                                                              \
    l0PingPong,                                                                                                \
    bl1ByPass,                                                                                                 \
    quantType,                                                                                                  \
    loadChannelType,                                                                                            \
    groupConvType,                                                                                             \
    outputOrder,                                                                                               \
    x,                                                                                                         \
    filter,                                                                                                    \
    bias,                                                                                                      \
    scale,                                                                                                     \
    offset,                                                                                                    \
    y,                                                                                                         \
    tilingData,                                                                                                \
    FuncImp)                                                                                                   \
    do {                                                                                                      \
        FuncImp<aConvType, bConvType,                                                                          \
                cConvType, biasConvType,                                                                       \
                Conv3DV2Param<l0PingPong, bl1ByPass, groupConvType, outputOrder, quantType, loadChannelType>> op;  \
        op.Init(x, filter, bias, scale, offset, y, user_workspace, &tilingData);                               \
        ASC_OP_LOGD("[Conv3dv2] Op init success.\n");                                                          \
        op.Process();                                                                                       \
        ASC_OP_LOGD("[Conv3dv2] Op process success.\n");                                                    \
    } while(0)

#define CONV3DV2_PROCESS(aConvType,                                                                            \
    bConvType,                                                                                                 \
    cConvType,                                                                                                 \
    biasConvType,                                                                                              \
    l0PingPong,                                                                                                \
    bl1ByPass,                                                                                                 \
    groupConvType,                                                                                             \
    outputOrder,                                                                                               \
    x,                                                                                                         \
    filter,                                                                                                    \
    bias,                                                                                                      \
    scale,                                                                                                     \
    offset,                                                                                                    \
    y,                                                                                                         \
    tilingData,                                                                                                \
    FuncImp)                                                                                                   \
    do {                                                                                                       \
        if (tilingData.conv3dApiTiling.quantType == 0) {                                                       \
            CONV3DV2_PROCESS_BASE(aConvType, bConvType, cConvType, biasConvType, l0PingPong, bl1ByPass,        \
                                  QuantType::NO_QUANT, LoadChannelType::NORMAL,groupConvType, outputOrder, x, filter, bias, scale,     \
                                  offset, y, tilingData, FuncImp);                                            \
        } else {                                                                                            \
            if (tilingData.conv3dApiTiling.scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_LC0))    \
            CONV3DV2_PROCESS_BASE(aConvType, bConvType, cConvType, biasConvType, l0PingPong, bl1ByPass,        \
                                  QuantType::PER_CHANNEL_NO_OFFSET, LoadChannelType::LOAD_TOTAL_LC0,groupConvType, outputOrder, x, filter,     \
                                  bias, scale, offset, y, tilingData, FuncImp);                                \
            else if (tilingData.conv3dApiTiling.scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_CORE))    \
            CONV3DV2_PROCESS_BASE(aConvType, bConvType, cConvType, biasConvType, l0PingPong, bl1ByPass,        \
                                QuantType::PER_CHANNEL_NO_OFFSET, LoadChannelType::LOAD_TOTAL_CORE,groupConvType, outputOrder, x, filter,     \
                                bias, scale, offset, y, tilingData, FuncImp);                                   \
            else if (tilingData.conv3dApiTiling.scaleAndBiasLoadType == static_cast<int8_t>(LoadChannelType::NORMAL))    \
            CONV3DV2_PROCESS_BASE(aConvType, bConvType, cConvType, biasConvType, l0PingPong, bl1ByPass,        \
                                QuantType::PER_CHANNEL_NO_OFFSET, LoadChannelType::NORMAL,groupConvType, outputOrder, x, filter,     \
                                bias, scale, offset, y, tilingData, FuncImp);                                  \
        }                                                                                                      \
    } while(0)

#define CONV3DV2_INIT_AND_PROCESS(aFormatP,                                                                    \
    bFormatP,                                                                                                  \
    cFormatP,                                                                                                  \
    aDataType,                                                                                                 \
    bDataType,                                                                                                 \
    cDataType,                                                                                                 \
    biasDataType,                                                                                              \
    l0PingPong,                                                                                                \
    bl1ByPass,                                                                                                 \
    groupConvType,                                                                                             \
    outputOrder,                                                                                               \
    x,                                                                                                         \
    filter,                                                                                                    \
    bias,                                                                                                      \
    scale,                                                                                                     \
    offset,                                                                                                    \
    y,                                                                                                         \
    tilingData)                                                                                                \
    do {                                                                                                       \
        using A_TYPE = ConvType<TPosition::GM, aFormatP, aDataType>;                                           \
        using B_TYPE = ConvType<TPosition::GM, bFormatP, bDataType>;                                           \
        using C_TYPE = ConvType<TPosition::GM, cFormatP, cDataType>;                                           \
        using BIAS_TYPE = ConvType<TPosition::GM, ConvFormat::ND, biasDataType>;                               \
        if constexpr (aFormat == ConvFormat::NCDHW) {                                                          \
            CONV3DV2_PROCESS(A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,                                                \
                            l0PingPong, bl1ByPass, groupConvType, outputOrder,                                 \
                            x, filter, bias, scale, offset, y, tilingData, KernelConv3DV2PointWise);           \
        } else {                                                                                               \
            CONV3DV2_PROCESS(A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,                                                \
                            l0PingPong, bl1ByPass, groupConvType, outputOrder,                                 \
                            x, filter, bias, scale, offset, y, tilingData, KernelConv3DV2);                    \
        }                                                                                                      \
    } while (0)

#define CONV3DV2_GROUPS_INIT_AND_PROCESS(aFormatP,                                                             \
    bFormatP,                                                                                                  \
    cFormatP,                                                                                                  \
    aDataType,                                                                                                 \
    bDataType,                                                                                                 \
    cDataType,                                                                                                 \
    biasDataType,                                                                                              \
    l0PingPong,                                                                                                \
    bl1ByPass,                                                                                                 \
    groupConvType,                                                                                             \
    outputOrder,                                                                                               \
    x,                                                                                                         \
    filter,                                                                                                    \
    bias,                                                                                                      \
    scale,                                                                                                     \
    offset,                                                                                                    \
    y,                                                                                                         \
    tilingData)                                                                                                \
    do {                                                                                                       \
        using A_TYPE = ConvType<TPosition::GM, aFormatP, aDataType>;                                           \
        using B_TYPE = ConvType<TPosition::GM, bFormatP, bDataType>;                                           \
        using C_TYPE = ConvType<TPosition::GM, cFormatP, cDataType>;                                           \
        using BIAS_TYPE = ConvType<TPosition::GM, ConvFormat::ND, biasDataType>;                               \
        CONV3DV2_PROCESS(A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,                                                    \
                        l0PingPong, bl1ByPass, groupConvType, outputOrder,                                     \
                        x, filter, bias, scale, offset, y, tilingData, KernelConv3DV2WithGroups);              \
    } while (0)

#define CONV3DV2_HW_MODE_INIT_AND_PROCESS(aFormatP,                                                             \
    bFormatP,                                                                                                  \
    cFormatP,                                                                                                  \
    aDataType,                                                                                                 \
    bDataType,                                                                                                 \
    cDataType,                                                                                                 \
    biasDataType,                                                                                              \
    l0PingPong,                                                                                                \
    bl1ByPass,                                                                                                 \
    groupConvType,                                                                                             \
    outputOrder,                                                                                               \
    x,                                                                                                         \
    filter,                                                                                                    \
    bias,                                                                                                      \
    scale,                                                                                                     \
    offset,                                                                                                    \
    y,                                                                                                         \
    tilingData)                                                                                                \
    do {                                                                                                       \
        using A_TYPE = ConvType<TPosition::GM, aFormatP, aDataType>;                                           \
        using B_TYPE = ConvType<TPosition::GM, bFormatP, bDataType>;                                           \
        using C_TYPE = ConvType<TPosition::GM, cFormatP, cDataType>;                                           \
        using BIAS_TYPE = ConvType<TPosition::GM, ConvFormat::ND, biasDataType>;                               \
        CONV3DV2_PROCESS(A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,                                                    \
                        l0PingPong, bl1ByPass, groupConvType, outputOrder,                                     \
                        x, filter, bias, scale, offset, y, tilingData, KernelConv3DV2HwMode);                  \
    } while (0)

extern "C" __global__ __aicore__ void conv3dv2(
            GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
            GM_ADDR offset_w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    ASC_OP_LOGD("[Conv3dv2] Begin to process conv3dv2.\n");
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR user_workspace = GetUserWorkspace(workspace);
    ASC_OP_LOGD("[Conv3dv2] Get workspace success.\n");
    GET_TILING_DATA(tilingData, tiling);
    ASC_OP_LOGD("[Conv3dv2] Get tiling data success.\n");
    ASC_OP_LOGD("[Conv3dv2] mUB %u, nUB %u, scaleandbiastype %u, quantType %u.\n",
                tilingData.conv3dApiTiling.mUB, tilingData.conv3dApiTiling.nUB,
                tilingData.conv3dApiTiling.scaleAndBiasLoadType, tilingData.conv3dApiTiling.quantType);

    if (TILING_KEY_IS(0)) {
        CONV3DV2_INIT_AND_PROCESS(aFormat,
            bFormat,
            cFormat,
            DTYPE_X,
            DTYPE_FILTER,
            DTYPE_Y,
            DTYPE_BIAS,
            ConvL0PingPong::ALL_CLOSE,
            bL1ByPassFlag,
            GroupConvType::NoGroup_Conv,
            OutputOrder::M_Mode,
            x,
            filter,
            bias,
            scale,
            offset,
            y,
            tilingData);
    } else if (TILING_KEY_IS(10)) {
        CONV3DV2_INIT_AND_PROCESS(aFormat,
            bFormat,
            cFormat,
            DTYPE_X,
            DTYPE_FILTER,
            DTYPE_Y,
            DTYPE_BIAS,
            ConvL0PingPong::ALL_CLOSE,
            bL1ByPassFlag,
            GroupConvType::NoGroup_Conv,
            OutputOrder::M_Mode,
            x,
            filter,
            bias,
            scale,
            offset,
            y,
            tilingData);
    } else if (TILING_KEY_IS(200)) {
        CONV3DV2_INIT_AND_PROCESS(aFormat,
            bFormat,
            cFormat,
            DTYPE_X,
            DTYPE_FILTER,
            DTYPE_Y,
            DTYPE_BIAS,
            ConvL0PingPong::L0A_OPEN,
            bL1ByPassFlag,
            GroupConvType::NoGroup_Conv,
            OutputOrder::M_Mode,
            x,
            filter,
            bias,
            scale,
            offset,
            y,
            tilingData);
    } else if (TILING_KEY_IS(210)) {
        CONV3DV2_INIT_AND_PROCESS(aFormat,
            bFormat,
            cFormat,
            DTYPE_X,
            DTYPE_FILTER,
            DTYPE_Y,
            DTYPE_BIAS,
            ConvL0PingPong::L0A_OPEN,
            bL1ByPassFlag,
            GroupConvType::NoGroup_Conv,
            OutputOrder::M_Mode,
            x,
            filter,
            bias,
            scale,
            offset,
            y,
            tilingData);
    } else if (TILING_KEY_IS(400)) {
        CONV3DV2_INIT_AND_PROCESS(aFormat,
            bFormat,
            cFormat,
            DTYPE_X,
            DTYPE_FILTER,
            DTYPE_Y,
            DTYPE_BIAS,
            ConvL0PingPong::L0B_OPEN,
            ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv,
            OutputOrder::M_Mode,
            x,
            filter,
            bias,
            scale,
            offset,
            y,
            tilingData);
    } else if (TILING_KEY_IS(410)) {
        CONV3DV2_INIT_AND_PROCESS(aFormat,
            bFormat,
            cFormat,
            DTYPE_X,
            DTYPE_FILTER,
            DTYPE_Y,
            DTYPE_BIAS,
            ConvL0PingPong::L0B_OPEN,
            bL1ByPassFlag,
            GroupConvType::NoGroup_Conv,
            OutputOrder::M_Mode,
            x,
            filter,
            bias,
            scale,
            offset,
            y,
            tilingData);
    } else if (TILING_KEY_IS(600)) {
        CONV3DV2_INIT_AND_PROCESS(aFormat,
            bFormat,
            cFormat,
            DTYPE_X,
            DTYPE_FILTER,
            DTYPE_Y,
            DTYPE_BIAS,
            ConvL0PingPong::ALL_OPEN,
            ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv,
            OutputOrder::M_Mode,
            x,
            filter,
            bias,
            scale,
            offset,
            y,
            tilingData);
    } else if (TILING_KEY_IS(610)) {
        CONV3DV2_INIT_AND_PROCESS(aFormat,
            bFormat,
            cFormat,
            DTYPE_X,
            DTYPE_FILTER,
            DTYPE_Y,
            DTYPE_BIAS,
            ConvL0PingPong::ALL_OPEN,
            bL1ByPassFlag,
            GroupConvType::NoGroup_Conv,
            OutputOrder::M_Mode,
            x,
            filter,
            bias,
            scale,
            offset,
            y,
            tilingData);
    }

    if constexpr (aFormat != ConvFormat::NCDHW) {
        if (TILING_KEY_IS(10010)) {
            CONV3DV2_GROUPS_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::ALL_CLOSE,
                bL1ByPassFlag,
                GroupConvType::GroupConv_Weight_Gfz,
                OutputOrder::M_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        } else if (TILING_KEY_IS(10210)) {
            CONV3DV2_GROUPS_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::L0A_OPEN,
                bL1ByPassFlag,
                GroupConvType::GroupConv_Weight_Gfz,
                OutputOrder::M_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        } else if (TILING_KEY_IS(10400)) {
            CONV3DV2_GROUPS_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::L0B_OPEN,
                ConvBL1ByPass::BYPASS_OFF,
                GroupConvType::GroupConv_Weight_Gfz,
                OutputOrder::M_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        } else if (TILING_KEY_IS(10410)) {
            CONV3DV2_GROUPS_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::L0B_OPEN,
                bL1ByPassFlag,
                GroupConvType::GroupConv_Weight_Gfz,
                OutputOrder::M_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        } else if (TILING_KEY_IS(10600)) {
            CONV3DV2_GROUPS_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::ALL_OPEN,
                ConvBL1ByPass::BYPASS_OFF,
                GroupConvType::GroupConv_Weight_Gfz,
                OutputOrder::M_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        } else if (TILING_KEY_IS(10610)) {
            CONV3DV2_GROUPS_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::ALL_OPEN,
                bL1ByPassFlag,
                GroupConvType::GroupConv_Weight_Gfz,
                OutputOrder::M_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        } else if (TILING_KEY_IS(100000)) {
            CONV3DV2_HW_MODE_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::ALL_CLOSE,
                bL1ByPassFlag,
                GroupConvType::NoGroup_Conv,
                OutputOrder::HW_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        } else if (TILING_KEY_IS(100010)) {
            CONV3DV2_HW_MODE_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::ALL_CLOSE,
                bL1ByPassFlag,
                GroupConvType::NoGroup_Conv,
                OutputOrder::HW_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        } else if (TILING_KEY_IS(100200)) {
            CONV3DV2_HW_MODE_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::L0A_OPEN,
                bL1ByPassFlag,
                GroupConvType::NoGroup_Conv,
                OutputOrder::HW_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        } else if (TILING_KEY_IS(100210)) {
            CONV3DV2_HW_MODE_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::L0A_OPEN,
                bL1ByPassFlag,
                GroupConvType::NoGroup_Conv,
                OutputOrder::HW_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        } else if (TILING_KEY_IS(100400)) {
            CONV3DV2_HW_MODE_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::L0B_OPEN,
                ConvBL1ByPass::BYPASS_OFF,
                GroupConvType::NoGroup_Conv,
                OutputOrder::HW_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        } else if (TILING_KEY_IS(100410)) {
            CONV3DV2_HW_MODE_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::L0B_OPEN,
                bL1ByPassFlag,
                GroupConvType::NoGroup_Conv,
                OutputOrder::HW_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        } else if (TILING_KEY_IS(100600)) {
            CONV3DV2_HW_MODE_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::ALL_OPEN,
                ConvBL1ByPass::BYPASS_OFF,
                GroupConvType::NoGroup_Conv,
                OutputOrder::HW_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        } else if (TILING_KEY_IS(100610)) {
            CONV3DV2_HW_MODE_INIT_AND_PROCESS(aFormat,
                bFormat,
                cFormat,
                DTYPE_X,
                DTYPE_FILTER,
                DTYPE_Y,
                DTYPE_BIAS,
                ConvL0PingPong::ALL_OPEN,
                bL1ByPassFlag,
                GroupConvType::NoGroup_Conv,
                OutputOrder::HW_Mode,
                x,
                filter,
                bias,
                scale,
                offset,
                y,
                tilingData);
        }
    }
    
    ASC_OP_LOGD("[Conv3dv2] End to process conv3dv2.\n");

    return;
}
