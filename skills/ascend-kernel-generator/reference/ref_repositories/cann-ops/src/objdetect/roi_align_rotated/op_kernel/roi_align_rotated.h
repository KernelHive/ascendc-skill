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
 * \file roi_align_rotated.h
 * \brief
 */
#ifndef _ROI_ALIGN_ROTATED_H_
#define _ROI_ALIGN_ROTATED_H_

#include <cmath>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "lib/matmul_intf.h"

namespace RoiAlignRotatedAll {
using namespace AscendC;
using namespace std;

constexpr uint32_t BUFFER_NUM = 2;
constexpr int32_t aligned_byte_num = 8;
constexpr int32_t aligned_data_num = 4;
constexpr int32_t rois_info_num = 6;
constexpr float one_value = 1.0f;
constexpr float negative_one_value = -1.0f;
constexpr float zero_value = 0.0f;
constexpr float half_value = -0.5f;

class RoiAlignRotated {
public:
    __aicore__ inline RoiAlignRotated()
    {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR rois, GM_ADDR output, const RoiAlignRotatedTilingData *tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        tileNum = tiling_data->tileNum;
        blockDim = tiling_data->blockDim;
        ubTotalSize = tiling_data->ub_total_size;
        batchSize = tiling_data->batch_size;
        channels = tiling_data->channels;
        channelsAligned = tiling_data->channels_aligned;
        inputH = tiling_data->input_h;
        inputW = tiling_data->input_w;
        roisNumAligned = tiling_data->rois_num_aligned;
        tailNum = tiling_data->tail_num;
        spatialScale = tiling_data->spatial_scale;
        samplingRatio = tiling_data->sampling_ratio;
        pooledHeight = tiling_data->pooled_height;
        pooledWidth = tiling_data->pooled_width;
        aligned = tiling_data->aligned;
        clockwise = tiling_data->clockwise;
        roisNumPerLcore = tiling_data->rois_num_per_Lcore;
        roisNumPerScore = tiling_data->rois_num_per_Score;
        lcoreNum = tiling_data->Lcore_num;
        scoreNum = tiling_data->Score_num;
        inputBufferSize = tiling_data->input_buffer_size;

        if (aligned == true) {
            offset = -0.5; // -0.5为对齐时的偏移量
        } else {
            offset = static_cast<float>(0);
        }

        total_rois_num = roisNumAligned - tailNum;
        output_shape = pooledHeight * pooledWidth;

        if (GetBlockIdx() < lcoreNum) {
            rois_num_per_core = roisNumPerLcore;
        } else {
            rois_num_per_core = roisNumPerScore;
        }

        uint32_t ub_size_for_loop = (static_cast<uint32_t>(ubTotalSize)) / aligned_byte_num;
        uint32_t roi_size = rois_info_num * sizeof(float);
        rois_num_per_loop_limit = ((ub_size_for_loop / roi_size) / aligned_byte_num) * aligned_byte_num;
        if (rois_num_per_core <= rois_num_per_loop_limit) {
            loopCount = 1;
            rois_num_per_loop = rois_num_per_core;
        } else {
            loopCount = (rois_num_per_core - rois_num_per_loop_limit) / rois_num_per_loop_limit + 1;
            rois_num_per_loop = rois_num_per_loop_limit;
        }

        uint32_t rois_buffer_size = rois_num_per_loop * sizeof(float);

        ASSERT(tileNum != 0 && "tile num can not be zero!");

        inputGM.SetGlobalBuffer((__gm__ float *)input, batchSize * channels * inputH * inputW);
        roisGM.SetGlobalBuffer((__gm__ float *)rois, (roisNumAligned * rois_info_num));
        outputGM.SetGlobalBuffer((__gm__ float *)output, (total_rois_num * channels * pooledHeight * pooledWidth));

        pipe.InitBuffer(roisQueueBatchIdx, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(roisQueueCenterX, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(roisQueueCenterY, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(roisQueueWidth, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(roisQueueHeight, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(roisQueueTheta, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(pwBuffer, rois_buffer_size);
        pipe.InitBuffer(phBuffer, rois_buffer_size);
        pipe.InitBuffer(binSizeHBuffer, rois_buffer_size);
        pipe.InitBuffer(binSizeWBuffer, rois_buffer_size);
        pipe.InitBuffer(binGridSizeHBuffer, rois_buffer_size);
        pipe.InitBuffer(binGridSizeWBuffer, rois_buffer_size);
        pipe.InitBuffer(gridHBuffer, rois_buffer_size);
        pipe.InitBuffer(gridWBuffer, rois_buffer_size);
        pipe.InitBuffer(sinThetaBuffer, rois_buffer_size);
        pipe.InitBuffer(cosThetaBuffer, rois_buffer_size);
        pipe.InitBuffer(roiStartHBuffer, rois_buffer_size);
        pipe.InitBuffer(roiStartWBuffer, rois_buffer_size);
        pipe.InitBuffer(countTensorBuffer, rois_buffer_size);
        pipe.InitBuffer(gridMulBuffer, rois_buffer_size);
        pipe.InitBuffer(outputValueBuffer, BUFFER_NUM, inputBufferSize);
        pipe.InitBuffer(countChannelBuffer, BUFFER_NUM, inputBufferSize);
        pipe.InitBuffer(weightBuffer, inputBufferSize);
        pipe.InitBuffer(valueBuffer, BUFFER_NUM, inputBufferSize);
        pipe.InitBuffer(tmpValueBuffer, inputBufferSize);
        pipe.InitBuffer(atomicAddBuffer, inputBufferSize);
        if (channels == channelsAligned) {
            pipe.InitBuffer(inQueueInput, BUFFER_NUM, inputBufferSize * aligned_data_num);
        } else {
            pipe.InitBuffer(inQueueInput, BUFFER_NUM, inputBufferSize);
        }
    }

    __aicore__ inline void Process()
    {
        if (rois_num_per_core > 0) {
            outputValue = outputValueBuffer.AllocTensor<float>();
            countChannel = countChannelBuffer.AllocTensor<float>();
            weightTensor = weightBuffer.Get<float>();
            valueTensor = valueBuffer.AllocTensor<float>();
            tmpValueTensor = tmpValueBuffer.Get<float>();

            binSizeH = binSizeHBuffer.Get<float>();
            binSizeW = binSizeWBuffer.Get<float>();
            roiBinGridH = binGridSizeHBuffer.Get<float>();
            roiBinGridW = binGridSizeWBuffer.Get<float>();
            gridHTensor = gridHBuffer.Get<float>();
            gridWTensor = gridWBuffer.Get<float>();
            roiSinTheta = sinThetaBuffer.Get<float>();
            roiCosTheta = cosThetaBuffer.Get<float>();
            roiStartH = roiStartHBuffer.Get<float>();
            roiStartW = roiStartWBuffer.Get<float>();
            countTensor = countTensorBuffer.Get<float>();
            gridMulTensor = gridMulBuffer.Get<float>();
            
            Pw = pwBuffer.Get<float>();
            Ph = phBuffer.Get<float>();
            atomicAddTensor = atomicAddBuffer.Get<float>();
            PipeBarrier<PIPE_V>();

            Duplicate(Pw, static_cast<float>(pooledWidth), rois_num_per_loop);
            Duplicate(Ph, static_cast<float>(pooledHeight), rois_num_per_loop);
            Duplicate(atomicAddTensor, one_value, channelsAligned);

            for (int32_t idx = channels; idx < channelsAligned; idx++) {
                atomicAddTensor.SetValue(idx, zero_value);
            }

            for (uint32_t i = 0; i < loopCount; i++) {
                RoisCopyIn(i, rois_num_per_loop);
                Compute(i, rois_num_per_loop);
            }

            outputValueBuffer.FreeTensor<float>(outputValue);
            countChannelBuffer.FreeTensor<float>(countChannel);
            valueBuffer.FreeTensor<float>(valueTensor);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void RoisCopyIn(uint32_t progress, int32_t rois_num)
    {
        LocalTensor<float> RoisBatchIdx = roisQueueBatchIdx.AllocTensor<float>();
        LocalTensor<float> RoisCenterX = roisQueueCenterX.AllocTensor<float>();
        LocalTensor<float> RoisCenterY = roisQueueCenterY.AllocTensor<float>();
        LocalTensor<float> RoisWidth = roisQueueWidth.AllocTensor<float>();
        LocalTensor<float> RoisHeight = roisQueueHeight.AllocTensor<float>();
        LocalTensor<float> RoisTheta = roisQueueTheta.AllocTensor<float>();
        PipeBarrier<PIPE_ALL>();

        if (GetBlockIdx() < lcoreNum) {
            int32_t pre_idx = GetBlockIdx() * rois_num_per_core + progress * rois_num_per_loop;
            DataCopy(RoisBatchIdx, roisGM[pre_idx], rois_num);
            DataCopy(RoisCenterX, roisGM[pre_idx + total_rois_num], rois_num);
            DataCopy(RoisCenterY, roisGM[pre_idx + total_rois_num * 2], rois_num); // RoisCenterY偏移量为pre_idx + total_rois_num * 2
            DataCopy(RoisWidth, roisGM[pre_idx + total_rois_num * 3], rois_num); // RoisWidth偏移量为pre_idx + total_rois_num * 3
            DataCopy(RoisHeight, roisGM[pre_idx + total_rois_num * 4], rois_num); // RoisHeight偏移量为pre_idx + total_rois_num * 4
            DataCopy(RoisTheta, roisGM[pre_idx + total_rois_num * 5], rois_num); // RoisTheta偏移量为pre_idx + total_rois_num * 5
            PipeBarrier<PIPE_ALL>();
        } else {
            int32_t pre_idx = lcoreNum * roisNumPerLcore + (GetBlockIdx() - lcoreNum) * rois_num_per_core + progress * rois_num_per_loop;
            DataCopy(RoisBatchIdx, roisGM[pre_idx], rois_num);
            DataCopy(RoisCenterX, roisGM[pre_idx + total_rois_num], rois_num);
            DataCopy(RoisCenterY, roisGM[pre_idx + total_rois_num * 2], rois_num); // RoisCenterY偏移量为pre_idx + total_rois_num * 2
            DataCopy(RoisWidth, roisGM[pre_idx + total_rois_num * 3], rois_num); // RoisWidth偏移量为pre_idx + total_rois_num * 3
            DataCopy(RoisHeight, roisGM[pre_idx + total_rois_num * 4], rois_num); // RoisHeight偏移量为pre_idx + total_rois_num * 4
            DataCopy(RoisTheta, roisGM[pre_idx + total_rois_num * 5], rois_num); // RoisTheta偏移量为pre_idx + total_rois_num * 5
            PipeBarrier<PIPE_ALL>();
        }
        
        roisQueueBatchIdx.EnQue<float>(RoisBatchIdx);
        roisQueueCenterX.EnQue<float>(RoisCenterX);
        roisQueueCenterY.EnQue<float>(RoisCenterY);
        roisQueueWidth.EnQue<float>(RoisWidth);
        roisQueueHeight.EnQue<float>(RoisHeight);
        roisQueueTheta.EnQue<float>(RoisTheta);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Compute(uint32_t progress, int32_t rois_num)
    {
        LocalTensor<float> RoisBatchIdx = roisQueueBatchIdx.DeQue<float>();
        LocalTensor<float> RoisCenterX = roisQueueCenterX.DeQue<float>();
        LocalTensor<float> RoisCenterY = roisQueueCenterY.DeQue<float>();
        LocalTensor<float> RoisWidth = roisQueueWidth.DeQue<float>();
        LocalTensor<float> RoisHeight = roisQueueHeight.DeQue<float>();
        LocalTensor<float> RoisTheta = roisQueueTheta.DeQue<float>();

        Muls(RoisCenterX, RoisCenterX, spatialScale, rois_num);
        Muls(RoisCenterY, RoisCenterY, spatialScale, rois_num);
        Muls(RoisWidth, RoisWidth, spatialScale, rois_num);
        Muls(RoisHeight, RoisHeight, spatialScale, rois_num);
        PipeBarrier<PIPE_V>();

        Adds(RoisCenterX, RoisCenterX, offset, rois_num);
        Adds(RoisCenterY, RoisCenterY, offset, rois_num);
        
        if (!aligned) {
            Maxs(RoisWidth, RoisWidth, one_value, rois_num);
            Maxs(RoisHeight, RoisHeight, one_value, rois_num);
        }
        
        if (clockwise) {
            Muls(RoisTheta, RoisTheta, negative_one_value, rois_num);
        }
        PipeBarrier<PIPE_V>();

        Muls(roiStartH, RoisHeight, half_value, rois_num);
        Muls(roiStartW, RoisWidth, half_value, rois_num);
        Div(binSizeH, RoisHeight, Ph, rois_num);
        Div(binSizeW, RoisWidth, Pw, rois_num);
        Sin(roiSinTheta, RoisTheta);
        Cos(roiCosTheta, RoisTheta);
        PipeBarrier<PIPE_V>();

        if (samplingRatio > 0) {
            roiBinGridH.SetSize(rois_num);
            Duplicate(roiBinGridH, static_cast<float>(samplingRatio), rois_num);
            Duplicate(roiBinGridW, static_cast<float>(samplingRatio), rois_num);
            PipeBarrier<PIPE_V>();
        } else {
            Ceil(roiBinGridH, binSizeH, rois_num);
            Ceil(roiBinGridW, binSizeW, rois_num);
            PipeBarrier<PIPE_V>();
        }
        
        Div(gridHTensor, binSizeH, roiBinGridH, rois_num);
        Div(gridWTensor, binSizeW, roiBinGridW, rois_num);
        Mul(gridMulTensor, roiBinGridW, roiBinGridH, rois_num);
        Maxs(countTensor, gridMulTensor, one_value, rois_num);
        PipeBarrier<PIPE_V>();

        int32_t output_index = ComputeOutputIndex(progress);

        for (uint32_t j = 0; j < rois_num; j++) {
            float batch_idx = RoisBatchIdx.GetValue(j);
            batch_idx = static_cast<int32_t>(batch_idx);

            if (output_index < (total_rois_num * pooledHeight * pooledWidth)) {
                ComputeItem(output_index, j, batch_idx, RoisCenterX.GetValue(j), RoisCenterY.GetValue(j));
            }
            output_index += output_shape;
        }
        
        roisQueueBatchIdx.FreeTensor<float>(RoisBatchIdx);
        roisQueueCenterX.FreeTensor<float>(RoisCenterX);
        roisQueueCenterY.FreeTensor<float>(RoisCenterY);
        roisQueueWidth.FreeTensor<float>(RoisWidth);
        roisQueueHeight.FreeTensor<float>(RoisHeight);
        roisQueueTheta.FreeTensor<float>(RoisTheta);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline float ComputeOutputIndex(uint32_t progress)
    {
        int32_t output_index;
        if (GetBlockIdx() < lcoreNum) {
            output_index = pooledHeight * pooledWidth * (GetBlockIdx() * rois_num_per_core + progress * rois_num_per_loop);
        } else {
            output_index = pooledHeight * pooledWidth * (lcoreNum * roisNumPerLcore + (GetBlockIdx() - lcoreNum) * rois_num_per_core + progress * rois_num_per_loop);
        }
        return output_index;
    }

    __aicore__ inline void ComputeItem(int32_t output_index, uint32_t roi_idx, int32_t batch_idx, float roi_center_w, float roi_center_h)
    {
        int32_t roi_bin_grid_h = roiBinGridH.GetValue(roi_idx);
        int32_t roi_bin_grid_w = roiBinGridW.GetValue(roi_idx);
        float bin_size_h = binSizeH.GetValue(roi_idx);
        float bin_size_w = binSizeW.GetValue(roi_idx);
        float grid_h = gridHTensor.GetValue(roi_idx);
        float grid_w = gridWTensor.GetValue(roi_idx);
        float roi_start_h = roiStartH.GetValue(roi_idx);
        float roi_start_w = roiStartW.GetValue(roi_idx);
        float sin_theta = roiSinTheta.GetValue(roi_idx);
        float cos_theta = roiCosTheta.GetValue(roi_idx);
        float count = countTensor.GetValue(roi_idx);
        Duplicate(countChannel, count, channelsAligned);
        
        for (int32_t index = output_index; index < (output_index + pooledHeight * pooledWidth); index++) {
            Duplicate(outputValue, zero_value, channelsAligned);
            int32_t pw = index / pooledWidth;
            int32_t ph = index / pooledWidth / pooledHeight;
            pw = index - pw * pooledWidth;
            ph = (index / pooledWidth) - ph * pooledHeight;
            for (uint32_t iy = 0; iy < roi_bin_grid_h; iy++) {
                const float yy = roi_start_h + ph * bin_size_h + (iy + .5f) * grid_h;
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    const float xx = roi_start_w + pw * bin_size_w + (ix + .5f) * grid_w;
                    float y = yy * cos_theta - xx * sin_theta + roi_center_h;
                    float x = yy * sin_theta + xx * cos_theta + roi_center_w;
                    bilinear_interpolate(batch_idx, x, y, index);
                    valueTensor = valueBuffer.DeQue<float>();
                    PipeBarrier<PIPE_ALL>();
                    Add(outputValue, outputValue, valueTensor, channels);
                    PipeBarrier<PIPE_V>();
                }
            }
            Div(outputValue, outputValue, countChannel, channels);
            PipeBarrier<PIPE_V>();
            if (channels != channelsAligned) {
                Mul(outputValue, outputValue, atomicAddTensor, channelsAligned);
                PipeBarrier<PIPE_V>();
                outputValueBuffer.EnQue<float>(outputValue);
                PipeBarrier<PIPE_ALL>();
                SetAtomicAdd<float>();
                SingleRoiCopyOut(index);
                SetAtomicNone();
            } else {
                outputValueBuffer.EnQue<float>(outputValue);
                PipeBarrier<PIPE_ALL>();
                if (index == (output_index + pooledHeight * pooledWidth - 2)) { // 常量2代表坐标值偏移量
                    Mul(outputValue, outputValue, atomicAddTensor, channelsAligned);
                    PipeBarrier<PIPE_V>();
                }
                SingleRoiCopyOut(index);
            }
        }
    }

    __aicore__ inline void bilinear_interpolate(int32_t batch_idx, float x, float y, int32_t index)
    {
        if (y <  (float)-1.0 or y > inputH or x < (float)-1.0 or x > inputW) {
            Duplicate(valueTensor, zero_value, channelsAligned);
            PipeBarrier<PIPE_ALL>();
        } else {
            if (y <= static_cast<float>(0)) {
                y = static_cast<float>(0);
            }
            if (x <= static_cast<float>(0)) {
                x = static_cast<float>(0);
            }
            int32_t x_floor = static_cast<int32_t>(x);
            int32_t y_floor = static_cast<int32_t>(y);
            int32_t x_ceil = x_floor + 1;
            int32_t y_ceil = y_floor + 1;  
            if (x_floor >= (inputW - 1)) {
                x_ceil = inputW - 1;
                x_floor = x_ceil;
                x = static_cast<float>(x_ceil);
            }
            if (y_floor >= inputH - 1) {
                y_ceil = inputH - 1;
                y_floor = y_ceil;
                y = static_cast<float>(y_ceil);
            }
            float lx = x - static_cast<float>(x_floor);
            float ly = y - static_cast<float>(y_floor);
            float hx = static_cast<float>(1) - lx;
            float hy = static_cast<float>(1) - ly;
            if ((channels == channelsAligned) and (x_ceil > x_floor) and (y_ceil > y_floor)) {
                AlignedBilinearInterpolate(batch_idx, hx, hy, lx, ly, x_floor, y_floor, x_ceil, y_ceil);
            } else {
                NonAlignedBilinearInterpolate(batch_idx, hx, hy, lx, ly, x_floor, y_floor, x_ceil, y_ceil);
            }
        }
        valueBuffer.EnQue<float>(valueTensor);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ void AlignedBilinearInterpolate(int32_t batch_idx, float hx, float hy, float lx, float ly, int32_t x_floor, int32_t y_floor, int32_t x_ceil, int32_t y_ceil)
    {
        LocalTensor<float> FeatureMap = inQueueInput.AllocTensor<float>();
        int32_t pre_idx = channels * (batch_idx * inputH * inputW);
        int32_t datacopy_idx = channels * (inputW * y_floor + x_floor) + pre_idx;
        PipeBarrier<PIPE_ALL>();
        AlignedSingleFeatureCopyIn(datacopy_idx, FeatureMap);
        FeatureMap = inQueueInput.DeQue<float>();
        float weight_p1 = hy * hx;
        float weight_p2 = hy * lx;
        float weight_p3 = ly * hx;
        float weight_p4 = ly * lx;
        Muls(tmpValueTensor, FeatureMap, weight_p1, channels);
        Muls(valueTensor, FeatureMap[channels], weight_p2, channels);
        PipeBarrier<PIPE_V>();
        Add(valueTensor, valueTensor, tmpValueTensor, channels);
        PipeBarrier<PIPE_V>();
        Muls(tmpValueTensor, FeatureMap[channels * 2], weight_p3, channels); // 起始偏移量为channels * 2
        PipeBarrier<PIPE_V>();
        Add(valueTensor, valueTensor, tmpValueTensor, channels);
        PipeBarrier<PIPE_V>();
        Muls(tmpValueTensor, FeatureMap[channels * 3], weight_p4, channels); // 起始偏移量为channels * 3
        PipeBarrier<PIPE_V>();
        Add(valueTensor, valueTensor, tmpValueTensor, channels);
        PipeBarrier<PIPE_V>();
        inQueueInput.FreeTensor<float>(FeatureMap);
    }

    __aicore__ void NonAlignedBilinearInterpolate(int32_t batch_idx, float hx, float hy, float lx, float ly, int32_t x_floor, int32_t y_floor, int32_t x_ceil, int32_t y_ceil)
    {
        LocalTensor<float> FeatureMap = inQueueInput.AllocTensor<float>();
        int32_t pre_idx = channels * (batch_idx * inputH * inputW);
        float weight = hy * hx;
        int32_t datacopy_idx = channels * (inputW * y_floor + x_floor) + pre_idx;
        PipeBarrier<PIPE_ALL>();
        NonAlignedSingleFeatureCopyIn(datacopy_idx, channelsAligned, FeatureMap);
        FeatureMap = inQueueInput.DeQue<float>();
        Muls(tmpValueTensor, FeatureMap, weight, channels);
        PipeBarrier<PIPE_ALL>();
        weight = hy * lx;
        datacopy_idx = channels * (inputW * y_floor + x_ceil) + pre_idx;
        NonAlignedSingleFeatureCopyIn(datacopy_idx, channelsAligned, FeatureMap);
        FeatureMap = inQueueInput.DeQue<float>();
        Muls(valueTensor, FeatureMap, weight, channels);
        PipeBarrier<PIPE_ALL>();
        weight = ly * hx;
        datacopy_idx = channels * (inputW * y_ceil + x_floor) + pre_idx;
        NonAlignedSingleFeatureCopyIn(datacopy_idx, channelsAligned, FeatureMap);
        FeatureMap = inQueueInput.DeQue<float>();
        Add(valueTensor, valueTensor, tmpValueTensor, channels);
        PipeBarrier<PIPE_ALL>();
        Muls(tmpValueTensor, FeatureMap, weight, channels);
        PipeBarrier<PIPE_ALL>();
        weight = ly * lx;
        datacopy_idx = channels * (inputW * y_ceil + x_ceil) + pre_idx;
        NonAlignedSingleFeatureCopyIn(datacopy_idx, channelsAligned, FeatureMap);
        FeatureMap = inQueueInput.DeQue<float>();
        Add(valueTensor, valueTensor, tmpValueTensor, channels);
        PipeBarrier<PIPE_ALL>();
        Muls(tmpValueTensor, FeatureMap, weight, channels);
        PipeBarrier<PIPE_ALL>();
        Add(valueTensor, valueTensor, tmpValueTensor, channels);
        PipeBarrier<PIPE_ALL>();
        inQueueInput.FreeTensor<float>(FeatureMap);
    }

    __aicore__ void AlignedSingleFeatureCopyIn(int32_t datacopy_idx, LocalTensor<float> FeatureMap)
    {
        DataCopyParams DataCopyParam = {
            (uint16_t)2, (uint16_t)(static_cast<uint16_t>(channels) * 2 / aligned_byte_num),
            (uint16_t)((static_cast<uint16_t>(channels) * (static_cast<uint16_t>(inputW) - 2)) / aligned_byte_num),
            (uint16_t)0
        };
        
        DataCopy(FeatureMap, inputGM[datacopy_idx], DataCopyParam);
        PipeBarrier<PIPE_ALL>();
        inQueueInput.EnQue<float>(FeatureMap);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ void NonAlignedSingleFeatureCopyIn(int32_t datacopy_idx, int32_t datacopy_len, LocalTensor<float> FeatureMap)
    {
        DataCopy(FeatureMap, inputGM[datacopy_idx], datacopy_len);
        PipeBarrier<PIPE_ALL>();
        inQueueInput.EnQue<float>(FeatureMap);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SingleRoiCopyOut(int32_t index)
    {
        outputValue = outputValueBuffer.DeQue<float>();
        PipeBarrier<PIPE_ALL>();
        DataCopy(outputGM[index * channels], outputValue, channelsAligned);
        PipeBarrier<PIPE_ALL>();
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> roisQueueBatchIdx;
    TQue<QuePosition::VECIN, BUFFER_NUM> roisQueueCenterX;
    TQue<QuePosition::VECIN, BUFFER_NUM> roisQueueCenterY;
    TQue<QuePosition::VECIN, BUFFER_NUM> roisQueueHeight;
    TQue<QuePosition::VECIN, BUFFER_NUM> roisQueueWidth;
    TQue<QuePosition::VECIN, BUFFER_NUM> roisQueueTheta;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput;
    TQue<QuePosition::VECOUT, BUFFER_NUM> countChannelBuffer;
    TQue<QuePosition::VECOUT, BUFFER_NUM> valueBuffer;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputValueBuffer;
    TBuf<TPosition::VECCALC> pwBuffer;
    TBuf<TPosition::VECCALC> phBuffer;
    TBuf<TPosition::VECCALC> weightBuffer;
    TBuf<TPosition::VECCALC> tmpValueBuffer;
    TBuf<TPosition::VECCALC> atomicAddBuffer;
    TBuf<TPosition::VECCALC> binSizeHBuffer;
    TBuf<TPosition::VECCALC> binSizeWBuffer;
    TBuf<TPosition::VECCALC> binGridSizeHBuffer;
    TBuf<TPosition::VECCALC> binGridSizeWBuffer;
    TBuf<TPosition::VECCALC> gridHBuffer;
    TBuf<TPosition::VECCALC> gridWBuffer;
    TBuf<TPosition::VECCALC> gridSizeHBuffer;
    TBuf<TPosition::VECCALC> sinThetaBuffer;
    TBuf<TPosition::VECCALC> cosThetaBuffer;
    TBuf<TPosition::VECCALC> roiStartHBuffer;
    TBuf<TPosition::VECCALC> roiStartWBuffer;
    TBuf<TPosition::VECCALC> countTensorBuffer;
    TBuf<TPosition::VECCALC> gridMulBuffer;

    GlobalTensor<float> inputGM;
    GlobalTensor<float> roisGM;
    GlobalTensor<float> outputGM;

    LocalTensor<float> Ph;
    LocalTensor<float> Pw;
    LocalTensor<float> binSizeH;
    LocalTensor<float> binSizeW;
    LocalTensor<float> roiBinGridH;
    LocalTensor<float> roiBinGridW;
    LocalTensor<float> gridHTensor;
    LocalTensor<float> gridWTensor;
    LocalTensor<float> roiSinTheta;
    LocalTensor<float> roiCosTheta;
    LocalTensor<float> roiStartH;
    LocalTensor<float> roiStartW;
    LocalTensor<float> countTensor;
    LocalTensor<float> gridMulTensor;
    LocalTensor<float> roiOutput;
    LocalTensor<float> outputValue;
    LocalTensor<float> countChannel;
    LocalTensor<float> weightTensor;
    LocalTensor<float> valueTensor;
    LocalTensor<float> tmpValueTensor;
    LocalTensor<float> atomicAddTensor;

    bool aligned;
    bool clockwise;
    uint32_t blockDim; 
    uint32_t tileNum;
    uint32_t batchSize;
    uint32_t channels;
    uint32_t channelsAligned;
    uint32_t inputH;
    uint32_t inputW;
    uint32_t roisNumAligned;
    uint32_t tailNum;
    uint32_t total_rois_num;
    uint32_t rois_num_per_core;
    uint32_t rois_num_per_loop;
    uint32_t rois_num_per_loop_limit;
    uint32_t loopCount;
    uint32_t roisNumPerLcore;
    uint32_t roisNumPerScore;
    uint32_t lcoreNum;
    uint32_t scoreNum;
    uint32_t inputBufferSize;
    int32_t samplingRatio;
    int32_t pooledHeight;
    int32_t pooledWidth;
    int32_t output_shape;
    float spatialScale;
    float offset;
    uint64_t ubTotalSize;
};
} // namespace RoiAlignRotatedAll

#endif // ROI_ALIGN_ROTATED_H