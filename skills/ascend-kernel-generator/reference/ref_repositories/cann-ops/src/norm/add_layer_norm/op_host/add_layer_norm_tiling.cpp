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
 * \file add_layer_norm_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "add_layer_norm_tiling.h"

namespace optiling {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_LOGI(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_LOGE(op_name, ...)            \
    std::printf(op_name, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_LOGW(op_name, ...)            \
    std::printf(op_name, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_CHECK(cond, log_func, return_expr) \
    if (cond) {                               \
        log_func;                             \
        return_expr;                          \
    }

#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)

}  // namespace optiling

namespace optiling {

static constexpr int UB_RESERVED_BYTE = 256;
static constexpr int BLOCK_SIZE = 32;
static constexpr int BIG_N_MAX_D = 500;
static constexpr int MAX_REPEAT_TIMES = 255;
static constexpr int FLOAT_BLOCK_ELEM = 8;
static constexpr int MAX_REPEAT_STRIDE_ELEM = 255 * FLOAT_BLOCK_ELEM;
static constexpr int CUBE_MAX_ELEM_FP32 = 16 * 16 / 2;
static constexpr int RESERVED_WORKSPACE_SIZE_910B = 16 * 1024 * 1024;
static constexpr int RESERVED_WORKSPACE_SIZE_310P = 2 * 1024 * 1024;
constexpr int MINIMUM_COLUMN_SLICE = 2048;

static constexpr uint32_t X1_DTYPE_OFFSET = 1000000;
static constexpr uint32_t X2_DTYPE_OFFSET = 100000;
static constexpr uint32_t GAMMA_DTYPE_OFFSET = 10000;
static constexpr int TILING_DATATYPE_FP16 = 1000;
static constexpr int TILING_DATATYPE_FP32 = 2000;
static constexpr int TILING_DATATYPE_BF16 = 3000;
static constexpr int TILING_ADDITIONAL_OUTPUT = 100;
static constexpr int TILING_SLICED = 10;
static constexpr int TILING_SINGLE_ROW = 20;
static constexpr int TILING_SINGLE_ROW_EXT = 30;
static constexpr int TILING_NORMAL_BIG_N = 40;
static constexpr int TILING_SLICE_EXT = 50;
static constexpr int TILING_NORMAL_SPECIAL = 60;
static constexpr int TILING_NORMAL_SPECIAL_REDUCE = 70;
static constexpr int TILING_NORMAL_BIG_N_SPECIAL_REDUCE = 80;
static constexpr int TILING_SINGLE_ROW_LESS_TENSOR = 90;
static constexpr int TILING_X3_PRESENT = 1;
static constexpr int TILING_X3_BROADCAST = 2;
constexpr int32_t INPUT_X1_INDEX = 0;
constexpr int32_t INPUT_X2_INDEX = 1;
constexpr int32_t INPUT_GAMMA_INDEX = 2;
constexpr int32_t INPUT_BETA_INDEX = 3;
constexpr int32_t OUTPUT_Y_INDEX = 0;
constexpr int32_t OUTPUT_MEAN_INDEX = 1;
constexpr int32_t OUTPUT_RSTD_INDEX = 2;
constexpr int32_t OUTPUT_X_INDEX = 3;
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t MIN_DIM_X = 1;
constexpr size_t MIN_DIM_GAMMA = 1;

const std::string OP_NAME = "AddLayerNorm";

inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

enum class TILING_TYPE {
    NORMAL,
    SLICE,
    SINGLE_ROW,
    SINGLE_ROW_EXT,
    NORMAL_BIG_N,
    SLICE_EXT,
    NORMAL_SPECIAL,
    NORMAL_SPECIAL_REDUCE,
    NORMAL_BIG_N_SPECIAL_REDUCE,
    SINGLE_ROW_LESS_TENSOR
};
enum class BIAS_TYPE { NO_BIAS, BROADCAST_BIAS, PRESENT_BIAS };
enum class DATA_TYPE { FP16 = 1, FP32 = 2, BF16 = 3 };

/**
 * TILING TRY WITH OPTIONS BUFFER_NUMBER AND ENABLE_X_OUT BY FOLLOW SEQUENCE:
 * 1. NORMAL CASE
 * 2. NORMAL CASE WITH SINGLE ROW
 * 2. NORMAL CASE WITH SINGLE ROW EXT
 * 3. SLICE REDUCE AXIS
 *
 * UB occupancy composition
 *  1. NORMAL CASE (COL=LAST_DIM, ROW=?)
 *   UB = DT*COUNT(X1,X2,Y,(ENABLE_X_OUT?X:NULL))*ROW*COL*BN + GAMMADT*COUNT(Gamma,Beta)*COL*BN +
 * DT*COUNT(BIAS_BROADCAST?BIAS:NULL)*COL*BN + DT_FLOAT*ROW*COUNT(Mean,Rstd)*BN +
 * DT_FLOAT*COUNT(x_buf,y_buf,z_buf)*ROW*COL + DT_FLOAT*64 + UB_RESERVED UB = DT*(3+(ENABLE_X_OUT?1:0))*ROW*LAST_DIM*BN
 * + GAMMADT*2*LAST_DIM*BN + 4*ROW*2*BN + 4*3*ROW*LAST_DIM + 4*64 + UB_RESERVED UB =
 * ROW*DT*(3+(ENABLE_X_OUT?1:0))*LAST_DIM*BN + GAMMADT*2*LAST_DIM*BN + ROW*4*2*BN + ROW*4*3*LAST_DIM + 4*64 +
 * UB_RESERVED UB = ROW*(DT*(3+(ENABLE_X_OUT?1:0))*LAST_DIM*BN + 4*2*BN + 4*3*LAST_DIM) + GAMMADT*2*LAST_DIM*BN + 4*64 +
 * UB_RESERVED ROW = (UB - (GAMMADT*2*LAST_DIM*BN + 4*64 + UB_RESERVED)) / (DT*(3+(ENABLE_X_OUT?1:0))*LAST_DIM*BN +
 * 4*2*BN + 4*3*LAST_DIM)
 *  2. NORMAL CASE WITH SINGLE ROW (ROW=1, COL=LAST_DIM)
 *   UB = DT*COUNT(X1,X2,Y,(ENABLE_X_OUT?X:NULL))*ROW*COL*BN + GAMMADT*COUNT(Gamma,Beta)*COL*BN +
 * DT_FLOAT*ROW*COUNT(Mean,Rstd)*BN + DT_FLOAT*COUNT(x_buf,y_buf)*COL + DT_FLOAT*64 + UB_RESERVED UB =
 * DT*(3+(ENABLE_X_OUT?1:0))*1*LAST_DIM*BN + GAMMADT*2*LAST_DIM*BN + DT_FLOAT*1*2*BN + DT_FLOAT*2*LAST_DIM + DT_FLOAT*64
 * + UB_RESERVED UB >? DT*(3+(ENABLE_X_OUT?1:0))*1*LAST_DIM*BN + GAMMADT*2*LAST_DIM*BN + DT_FLOAT*1*2*BN +
 * DT_FLOAT*2*LAST_DIM + DT_FLOAT*64 + UB_RESERVED
 *  3. NORMAL CASE WITH SINGLE ROW EXT SHARE X/Y VECIN/OUT QUEUE (ROW=1, COL=LAST_DIM)
 *   UB = DT*COUNT(X1,Y)*ROW*COL*BN + GAMMADT*COUNT(Gamma,Beta)*COL*BN + DT_FLOAT*ROW*COUNT(Mean,Rstd)*BN +
 * DT_FLOAT*COUNT(x_buf,y_buf)*COL + DT_FLOAT*64 + UB_RESERVED UB = DT*(2)*1*LAST_DIM*BN + GAMMADT*2*LAST_DIM*BN +
 * DT_FLOAT*1*2*BN + DT_FLOAT*2*LAST_DIM + DT_FLOAT*64 + UB_RESERVED UB >? DT*(2)*1*LAST_DIM*BN + GAMMADT*2*LAST_DIM*BN
 * + DT_FLOAT*1*2*BN + DT_FLOAT*2*LAST_DIM + DT_FLOAT*64 + UB_RESERVED
 *  4. SLICE REDUCE AXIS WITH FULL LOAD GAMMA/BETA/X (ROW=1, COL=?)
 *   UB = DT*COUNT(X1,X2,Y,(ENABLE_X_OUT?X:NULL))*ROW*COL*BN + GAMMADT*COUNT(Gamma,Beta)*LAST_DIM*BN +
 * DT*COUNT(BIAS_BROADCAST?BIAS:NULL)*LAST_DIM*BN + DT_FLOAT*ROW*COUNT(Mean,Rstd)*BN + DT_FLOAT*COUNT(x_buf)*LAST_DIM +
 * DT_FLOAT*COUNT(y_buf,z_buf)*COL + DT_FLOAT*64 + DT_FLOAT*COUNT(mean_buf,rstd_buf) + UB_RESERVED UB =
 * DT*(3+(ENABLE_X_OUT?1:0))*1*COL*BN + GAMMADT*2*LAST_DIM*BN + DT_FLOAT*1*2*BN + DT_FLOAT*1*LAST_DIM + DT_FLOAT*2*COL +
 * DT_FLOAT*64 + DT_FLOAT*2 + UB_RESERVED UB = COL*DT*(3+(ENABLE_X_OUT?1:0))*1*BN + GAMMADT*2*LAST_DIM*BN +
 * DT_FLOAT*1*2*BN + DT_FLOAT*1*LAST_DIM + COL*DT_FLOAT*2 + DT_FLOAT*64 + DT_FLOAT*2 + UB_RESERVED UB =
 * COL*(DT*(3+(ENABLE_X_OUT?1:0))*1*BN + DT_FLOAT*2) + GAMMADT*2*LAST_DIM*BN + DT_FLOAT*1*2*BN + DT_FLOAT*1*LAST_DIM +
 * DT_FLOAT*64 + DT_FLOAT*2 + UB_RESERVED COL = (UB - (GAMMADT*2*LAST_DIM*BN + DT_FLOAT*1*2*BN + DT_FLOAT*1*LAST_DIM +
 * DT_FLOAT*64 + DT_FLOAT*2 + DT_FLOAT*64 + UB_RESERVED)) / (DT*(3+(ENABLE_X_OUT?1:0))*1*BN + DT_FLOAT*2)
 *  5. SLICE EXT (ROW=1, COL=?)
 *   UB = DT*COUNT(X1,X2,Y,(ENABLE_X_OUT?X:NULL))*ROW*COL*BN + GAMMADT*COUNT(Gamma,Beta)*COL*BN +
 * DT*COUNT(BIAS_BROADCAST?BIAS:NULL)*COL*BN + DT_FLOAT*ROW*COUNT(Mean,Rstd)*BN + DT_FLOAT*COUNT(x_buf,y_buf,z_buf)*COL
 * + DT_FLOAT*64 + UB_RESERVED UB = DT*(3+(ENABLE_X_OUT?1:0))*1*COL*BN + GAMMADT*2*COL*BN +
 * DT*(BIAS_BROADCAST?1:0)*COL*BN + DT_FLOAT*1*2*BN + DT_FLOAT*3*COL + DT_FLOAT*64 + UB_RESERVED UB =
 * COL*DT*(3+(ENABLE_X_OUT?1:0))*1*BN + COL*(GAMMADT*2*BN + DT*(BIAS_BROADCAST?1:0)*BN) + DT_FLOAT*1*2*BN +
 * COL*DT_FLOAT*3 + DT_FLOAT*64 + UB_RESERVED UB = COL*(DT*(3+(ENABLE_X_OUT?1:0))*1*BN + GAMMADT*(2*BN) +
 * DT*(BIAS_BROADCAST?1:0)*BN + DT_FLOAT*3) + DT_FLOAT*1*2*BN + DT_FLOAT*64 + UB_RESERVED COL = (UB - (DT_FLOAT*1*2*BN +
 * DT_FLOAT*64 + UB_RESERVED)) / (DT*(3+(ENABLE_X_OUT?1:0))*1*BN + GAMMADT*2*BN + DT*(BIAS_BROADCAST?1:0)*BN +
 * DT_FLOAT*3)
 */
inline TILING_TYPE AddLayerNormTilingImpl(uint64_t maxUbSize, ge::DataType dataType, ge::DataType betagammaDataType,
    int32_t bufferNum, int32_t numCol, bool enableXOut, enum BIAS_TYPE biasType, uint32_t &rowPerTime,
    uint32_t &colPerTime, bool is310P)
{
    int dtSize = GetSizeByDataType(dataType);
    int betagammaDtSize = GetSizeByDataType(betagammaDataType);
    auto numColAligned = (numCol + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    bool isAllInputFP16 = (dataType == ge::DT_FLOAT16) && (betagammaDataType == ge::DT_FLOAT16);
    bool isAllInputFP32 = (dataType == ge::DT_FLOAT) && (betagammaDataType == ge::DT_FLOAT);
    bool isXB16GammaB32 =
        ((dataType == ge::DT_FLOAT16) || (dataType == ge::DT_BF16)) && (betagammaDataType == ge::DT_FLOAT);
    bool isNoBias = (biasType == BIAS_TYPE::NO_BIAS);
    bool isBroadcastBias = (biasType == BIAS_TYPE::BROADCAST_BIAS);
    int biasNum = isBroadcastBias ? 1 : 0;
    int additionalOutputNum = enableXOut ? 1 : 0;

    uint32_t blockElementNum = FLOAT_BLOCK_ELEM;
    if (dtSize > 0) {
        blockElementNum = BLOCK_SIZE / dtSize;
    }
    // try 310 normal tiling case (with special reducesum and broadcast implementation):
    auto numColElemAligned = (numCol + blockElementNum - 1) / blockElementNum * blockElementNum;
    bool isSpecialReduceTiling = is310P && enableXOut && isNoBias && (dataType == ge::DT_FLOAT16) &&
                                 ((numCol % blockElementNum) == 0) && (numColElemAligned <= MAX_REPEAT_STRIDE_ELEM);
    if (isSpecialReduceTiling) {
        // Tensor(n_dim != 1): x1(dtype)/x2(dtype)/y(dtype)/x_que(dtype)/x(fp32)/z(fp32)/y(fp32)
        // Tensor(n_dim == 1): beta(float32)/gamma(float32)/bias(dtype)
        // other Tensor: transposeSrcBuf(int16)/transposeDstBuf(int16)/orBuf(int16)
        auto betaGammaBufSize = sizeof(float) * 2 * numColAligned * bufferNum;
        auto preRowInputOutputBufSize = dtSize * 3 * numColAligned * bufferNum;  // x1/x2/y
        auto preRowTmpBufSize = sizeof(float) * 2 * numColAligned;               // x_buf/z_buf
        auto preRowReduceBufSize = sizeof(float) * 64;                           // y_buf
        auto transposeBufSize =
            sizeof(int16_t) * 2 * 16 * 16 * 8;  // transposeSrcBuf/transposeDstBuf shape=16*16*8 fp16
        auto orBufSize = sizeof(int16_t) * 16;  // orBufINT16 size is one block
        auto tmpRow = static_cast<double>(static_cast<long>(maxUbSize) -
                                          (betaGammaBufSize + UB_RESERVED_BYTE + transposeBufSize + orBufSize)) /
                      static_cast<double>(preRowInputOutputBufSize + preRowTmpBufSize + preRowReduceBufSize);
        bool enouthForSpecialReduce = tmpRow > 1 && tmpRow <= CUBE_MAX_ELEM_FP32;
        if (enouthForSpecialReduce) {
            rowPerTime = floor(tmpRow);
            colPerTime = numCol;
            return TILING_TYPE::NORMAL_SPECIAL_REDUCE;
        }
    }
    // try 910 normal tiling case (with special reducesum and broadcast implementation):
    isSpecialReduceTiling =
        (!is310P) && (isNoBias || isBroadcastBias) && isAllInputFP16 && (numColElemAligned <= MAX_REPEAT_STRIDE_ELEM);
    if (isSpecialReduceTiling) {
        // Tensor(n_dim != 1): x1(dtype)/x2(dtype)/y(dtype)/mean(fp32)/rstd(fp32)/x_que(dtype)/x(fp32)/z(fp32)/y(fp32)
        // Tensor(n_dim == 1): beta(float32)/gamma(float32)/bias(dtype)
        auto betaGammaBufSize = sizeof(float) * 2 * numColAligned * bufferNum;
        auto biasBufSize = dtSize * biasNum * numColAligned * bufferNum;
        auto preRowInputOutputBufSize =
            dtSize * 3 * numColAligned * bufferNum + sizeof(float) * 2 * bufferNum;  // x1/x2/y/mean/rstd
        auto preRowTmpBufSize = sizeof(float) * 2 * numColAligned;                   // x_buf/z_buf
        auto preRowReduceBufSize = sizeof(float) * 64;                               // y_buf
        auto brcbBufSize = sizeof(float) * 64 * 8;                                   // brcb need 8 elem align
        auto tmpRow = static_cast<double>(static_cast<long>(maxUbSize) -
                                          (betaGammaBufSize + biasBufSize + UB_RESERVED_BYTE + brcbBufSize)) /
                      static_cast<double>(preRowInputOutputBufSize + preRowTmpBufSize + preRowReduceBufSize);
        if (tmpRow > 1) {
            rowPerTime = floor(tmpRow);
            colPerTime = numCol;
            return TILING_TYPE::NORMAL_SPECIAL_REDUCE;
        }
    }
    // try 910 normal tiling case:
    auto tmpRow = static_cast<double>(static_cast<long>(maxUbSize) -
                                      (betagammaDtSize * 2 * numColAligned * bufferNum +
                                          dtSize * biasNum * numColAligned * bufferNum + 4 * 64 + UB_RESERVED_BYTE)) /
                  static_cast<double>(dtSize * (3 + additionalOutputNum) * numColAligned * bufferNum +
                                      4 * 2 * bufferNum + 4 * 3 * numColAligned);
    // try normal tiling special case, 2 is fp16 or bf16 dtype size
    bool isSpecialTiling = (!is310P) && (isBroadcastBias) && isAllInputFP16;
    // beta, gamma, bias on ub, count(beta, gamma) = 2, mul(4, 64) required by mean and rstd
    int32_t ubCacheSize =
        sizeof(float) * 2 * numColAligned * bufferNum + dtSize * numColAligned * bufferNum + 4 * 64 + UB_RESERVED_BYTE;
    // ubsize for one row compute, we have 3 TQue and 2 fp32 TBuf, and count(mean, rstd) = 2 TQues, which size is 4
    // Bytes
    int32_t preRowSize =
        dtSize * 3 * numColAligned * bufferNum + sizeof(float) * 2 * bufferNum + sizeof(float) * 2 * numColAligned;
    double tmpRowSpecial = (static_cast<double>(maxUbSize) - (double)ubCacheSize) / ((double)preRowSize);
    // apply BetterUB opt when Normal tiling process 1.x rows per loop and Normal Special process 2.0 rows per loop.
    bool normalSpecialTiling = isSpecialTiling && tmpRow < 2.0 && tmpRow > 0.0 && tmpRowSpecial > 2.0;
    if (normalSpecialTiling) {
        rowPerTime = floor(tmpRowSpecial);
        colPerTime = numCol;
        return TILING_TYPE::NORMAL_SPECIAL;
    } else if (tmpRow > 1) {
        rowPerTime = floor(tmpRow);
        colPerTime = numCol;
        return TILING_TYPE::NORMAL;
    }
    // try second case
    // 3 TQue.
    int64_t singleRowQueSize = dtSize * (3 + additionalOutputNum) * 1 * static_cast<int64_t>(numCol) * bufferNum;
    // 2 tmp TBuf + 2 weight TBuf, fp32 dtsize is 4.
    int64_t singleRowBufSize = (4 * 2 + betagammaDtSize * 2 * bufferNum) * static_cast<int64_t>(numCol);
    // 64 for rep elements, fp32 dtsize is 4.
    int64_t singleRowConstSize = static_cast<int64_t>(4 * 1 * 2 * bufferNum + 4 * 64 + UB_RESERVED_BYTE);
    uint64_t tmpUbSize = static_cast<uint64_t>(singleRowQueSize + singleRowBufSize + singleRowConstSize);
    if (tmpUbSize < maxUbSize) {
        rowPerTime = 1;
        colPerTime = numCol;
        return TILING_TYPE::SINGLE_ROW;
    }
    // try SINGLE ROW share X vec in queue
    // 2 TQue.
    singleRowQueSize =
        static_cast<int64_t>(dtSize) * 2 * 1 * static_cast<int64_t>(numCol) * static_cast<int64_t>(bufferNum);
    // 2 tmp TBuf + 2 weight TBuf, fp32 dtsize is 4.
    singleRowBufSize = static_cast<int64_t>(4 * 2 + betagammaDtSize * 2 * bufferNum) * static_cast<int64_t>(numCol);
    // 2 tmp TBuf + 2 weight TBuf, fp32 dtsize is 4， 64 is tmpBuffer size for compute.
    singleRowConstSize = static_cast<int64_t>(4 * 1 * 2 * bufferNum + 4 * 64 + UB_RESERVED_BYTE);
    tmpUbSize = static_cast<uint64_t>(singleRowQueSize + singleRowBufSize + singleRowConstSize);
    if (tmpUbSize < maxUbSize) {
        rowPerTime = 1;
        colPerTime = numCol;
        return TILING_TYPE::SINGLE_ROW_EXT;
    }

    // try SINGLE ROW LESS TENSOR
    // use less tensor size to cover larger last dim case.
    bool isSingleRowLessTensorTiling = (!is310P) && isNoBias && isAllInputFP32 && enableXOut;
    if (isSingleRowLessTensorTiling) {
        auto oneRowX1X2YBufSize = dtSize * 1 * numCol * bufferNum;
        auto oneRowMeanRstdBufSize = 4 * 1 * 2 * bufferNum;
        auto oneRowTmpBufSize = 4 * 2 * numCol;
        auto oneRowReduceBufSize = 4 * 64;
        tmpUbSize =
            oneRowX1X2YBufSize + oneRowMeanRstdBufSize + oneRowTmpBufSize + oneRowReduceBufSize + UB_RESERVED_BYTE;
        if (tmpUbSize < maxUbSize) {
            rowPerTime = 1;
            colPerTime = numCol;
            return TILING_TYPE::SINGLE_ROW_LESS_TENSOR;
        }
    }

    // when x1/x2: fp16/bf16 beta/gamma:fp32.
    bool isSingleRowLessTensor2 = (!is310P) && (isNoBias || isBroadcastBias) && isXB16GammaB32 && enableXOut;
    if (isSingleRowLessTensor2) {
        auto oneRowX1X2YBufSize = dtSize * 1 * numCol + 4 * 1 * numCol;
        auto oneRowMeanRstdBufSize = 4 * 1 * 2;
        auto oneRowTmpBufSize = 4 * 2 * numCol;
        auto oneRowReduceBufSize = 4 * 64;
        tmpUbSize =
            oneRowX1X2YBufSize + oneRowMeanRstdBufSize + oneRowTmpBufSize + oneRowReduceBufSize + UB_RESERVED_BYTE;
        if (tmpUbSize < maxUbSize) {
            rowPerTime = 1;
            colPerTime = numCol;
            return TILING_TYPE::SINGLE_ROW_LESS_TENSOR;
        }
    }

    // try third case
    int numPerBlock = 1;
    if (dtSize > 0) {
        numPerBlock = BLOCK_SIZE / dtSize;
    }
    rowPerTime = 1;

    // 3 tmp TBuf + fp32 dtsize is 4.
    int64_t ubColNumSlice = static_cast<int64_t>(dtSize * (3 + additionalOutputNum) * bufferNum + 4 * 2);
    // 2 weight Buf, 1 bias Buf, fp32 dtsize is 4, rep element for fp 32 is 64.
    int64_t ubConstSlice = static_cast<int64_t>(betagammaDtSize) * 2 * static_cast<int64_t>(numCol) +
                           static_cast<int64_t>(dtSize * biasNum * bufferNum) * static_cast<int64_t>(numCol) +
                           static_cast<int64_t>(dtSize * 2 * bufferNum) + 4 * static_cast<int64_t>(numCol) +
                           static_cast<int64_t>(4 * 64 + UB_RESERVED_BYTE);
    double tmpCol =
        static_cast<double>(static_cast<int64_t>(maxUbSize) - ubConstSlice) / static_cast<double>(ubColNumSlice);
    if (tmpCol > MINIMUM_COLUMN_SLICE) {
        colPerTime = floor(tmpCol);
        if (colPerTime % numPerBlock != 0) {  // colPerTime should be 32 align
            colPerTime = (colPerTime / numPerBlock) * numPerBlock;
            return TILING_TYPE::SLICE;
        }
    }

    rowPerTime = 1;
    // 2 reduce Buf, fp32 dtsize is 4, rep element for fp 32 is 64.
    ubConstSlice = 4 * 2 * bufferNum + 4 * 64 + UB_RESERVED_BYTE;
    // 3 TQues, 2 TBuf for gammabeta, 3 fp32 TBufs, sizeof fp32 is 4.
    ubColNumSlice = dtSize * (3 + additionalOutputNum) * bufferNum + betagammaDtSize * 2 * bufferNum +
                    dtSize * biasNum * bufferNum + 4 * 3;
    tmpCol = static_cast<double>(static_cast<int64_t>(maxUbSize) - ubConstSlice) / static_cast<double>(ubColNumSlice);
    colPerTime = floor(tmpCol);
    if (colPerTime % numPerBlock != 0) {  // colPerTime should be 32 align
        colPerTime = (colPerTime / numPerBlock) * numPerBlock;
    }
    return TILING_TYPE::SLICE_EXT;
}

static bool CheckDimLimit(size_t x1DimNum, size_t gammaDimNum)
{
    OP_TILING_CHECK(x1DimNum > MAX_DIM_NUM || x1DimNum < MIN_DIM_X,
        OP_LOGE("CheckDimLimit", "Input x1's dim num should not greater than 8 or smaller than 1."),
        return false);
    OP_TILING_CHECK(gammaDimNum > MAX_DIM_NUM || gammaDimNum < MIN_DIM_GAMMA,
        OP_LOGE("CheckDimLimit", "Input gamma's dim num should not greater than 8 or smaller than 1."),
        return false);
    return true;
}

template <typename T>
static bool CheckNotNullAll(std::initializer_list<T> ptrList)
{
    for (auto &ptr : ptrList) {
        if (ptr == nullptr) {
            return false;
        }
    }
    return true;
}

template <typename T>
static inline bool CheckEqualsAll(std::initializer_list<T> eleList)
{
    bool ret = true;
    if (eleList.size() > 0) {
        const T *fontPtr = eleList.begin();
        for (const T *curtPtr = eleList.begin(); curtPtr != eleList.end(); curtPtr++) {
            ret = ret && (*(curtPtr) == *(fontPtr));
            fontPtr = curtPtr;
        }
    }
    return ret;
}

static inline bool HasNoZero(const gert::StorageShape *shapePtr, size_t shapeDim)
{
    for (uint32_t i = 0; i < shapeDim; i++) {
        OP_TILING_CHECK(shapePtr->GetStorageShape().GetDim(i) == 0,
            OP_LOGE("HasNoZero", "Invaild shape which have zero dim."),
            return false);
    }
    return true;
}

static inline bool CheckX1GammaMean(const gert::StorageShape *x1Shape, const gert::StorageShape *gammaShape,
    const gert::StorageShape *meanShape, size_t x1DimNum, size_t gammaDimNum)
{
    for (uint32_t i = 0; i < gammaDimNum; i++) {
        OP_TILING_CHECK(
            gammaShape->GetStorageShape().GetDim(i) != x1Shape->GetStorageShape().GetDim(x1DimNum - gammaDimNum + i),
            OP_LOGE("CheckAddLn", "Gamma shape invaild, gamma shape is not equal x1 last few dim."),
            return false);
    }
    for (uint32_t i = 0; i < x1DimNum; i++) {
        if (i < x1DimNum - gammaDimNum) {
            OP_TILING_CHECK(meanShape->GetStorageShape().GetDim(i) != x1Shape->GetStorageShape().GetDim(i),
                OP_LOGE("CheckAddLn", "Output mean/rstd reduce dim is not equal x1 first few dim."),
                return false);
        } else {
            OP_TILING_CHECK(meanShape->GetStorageShape().GetDim(i) != 1,
                OP_LOGE("CheckAddLn", "Output mean/rstd reduce dim is not equal 1."),
                return false);
        }
    }
    return true;
}

static bool CheckInputShape4AddLayerNorm(const gert::TilingContext *context)
{
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape *betaShape = context->GetInputShape(INPUT_BETA_INDEX);
    const gert::StorageShape *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    const gert::StorageShape *meanShape = context->GetOutputShape(OUTPUT_MEAN_INDEX);
    const gert::StorageShape *rstdShape = context->GetOutputShape(OUTPUT_RSTD_INDEX);
    const gert::StorageShape *xShape = context->GetOutputShape(OUTPUT_X_INDEX);
    bool noNull = CheckNotNullAll({x1Shape, x2Shape, gammaShape, betaShape, yShape, meanShape, rstdShape, xShape});
    OP_TILING_CHECK(!noNull, OP_LOGE("CheckAddLn", "In/Out have Null."), return false);
    size_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t betaDimNum = betaShape->GetStorageShape().GetDimNum();
    size_t yDimNum = yShape->GetStorageShape().GetDimNum();
    size_t meanDimNum = meanShape->GetStorageShape().GetDimNum();
    size_t rstdDimNum = rstdShape->GetStorageShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    OP_LOGI("CheckAddLn",
        "x1 DimNum = %zu, x2 DimNum = %zu, gamma DimNum = %zu, beta DimNum = %zu.",
        x1DimNum,
        x2DimNum,
        gammaDimNum,
        betaDimNum);
    OP_LOGI("CheckAddLn",
        "y DimNum = %zu, mean DimNum = %zu, rstd DimNum = %zu, x DimNum = %zu.",
        yDimNum,
        meanDimNum,
        rstdDimNum,
        xDimNum);
    OP_TILING_CHECK(!CheckDimLimit(x1DimNum, gammaDimNum), OP_LOGE("CheckAddLn", "Bad Input DimNum."), return false);
    OP_TILING_CHECK(!CheckEqualsAll<size_t>({x1DimNum, x2DimNum, yDimNum, xDimNum, meanDimNum, rstdDimNum}),
        OP_LOGE("CheckAddLn", "Input x2 shape invaild, dim num is not equal x1 dim."),
        return false);
    OP_TILING_CHECK((x1DimNum <= 0 || gammaDimNum <= 0 || betaDimNum <= 0),
        OP_LOGE("CheckAddLn", "Input x1/x2/gamma/beta shape should not be smaller or equal to zero."),
        return false);
    OP_TILING_CHECK(x1DimNum < gammaDimNum,
        OP_LOGE("CheckAddLn", "x1 dim num should not be smaller than gamma dim num."),
        return false);
    OP_TILING_CHECK(!HasNoZero(x1Shape, x1DimNum), OP_LOGE("CheckAddLn", "Shape x1 have zero dim."), return false);
    OP_TILING_CHECK(
        !HasNoZero(gammaShape, gammaDimNum), OP_LOGE("CheckAddLn", "Shape gamma have zero dim."), return false);
    OP_TILING_CHECK(
        !HasNoZero(meanShape, meanDimNum), OP_LOGE("CheckAddLn", "Shape mean have zero dim."), return false);
    bool equalX1Shape = CheckEqualsAll<const gert::StorageShape>({*(x1Shape), *(x2Shape), *(yShape), *(xShape)});
    bool equalGammaShape = CheckEqualsAll<const gert::StorageShape>({*(gammaShape), *(betaShape)});
    bool equalReduceShape = CheckEqualsAll<const gert::StorageShape>({*(meanShape), *(rstdShape)});
    OP_TILING_CHECK(!equalX1Shape, OP_LOGE("CheckAddLn", "Shape x1/x2/x/y Not same."), return false);
    OP_TILING_CHECK(!equalGammaShape, OP_LOGE("CheckAddLn", "Shape gamma/beta Not same."), return false);
    OP_TILING_CHECK(!equalReduceShape, OP_LOGE("CheckAddLn", "Shape mean/rstd Not same."), return false);
    OP_TILING_CHECK(!CheckX1GammaMean(x1Shape, gammaShape, meanShape, x1DimNum, gammaDimNum),
        OP_LOGE("CheckAddLn", "Shape x1/gamma/mean check failed."),
        return false);
    return true;
}

static ge::graphStatus Tiling4AddLayerNorm(gert::TilingContext *context)
{
    OP_TILING_CHECK(!CheckInputShape4AddLayerNorm(context),
        OP_LOGE(context->GetNodeName(), "Input shape invalid."),
        return ge::GRAPH_FAILED);
    AddLayerNormTilingData tiling;
    float eps = *context->GetAttrs()->GetFloat(0);
    auto enableAdditionalOutput = *context->GetAttrs()->GetBool(1);
    tiling.set_eps(eps);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    bool is310P = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P ? true : false;
    uint64_t maxUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);
    auto maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    size_t xDimNum = context->GetInputShape(INPUT_X1_INDEX)->GetStorageShape().GetDimNum();
    size_t gammaDimNum = context->GetInputShape(INPUT_GAMMA_INDEX)->GetStorageShape().GetDimNum();
    int32_t numRow = 1;
    int32_t numCol = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= context->GetInputShape(INPUT_X1_INDEX)->GetStorageShape().GetDim(i);
    }
    for (size_t i = 0; i < gammaDimNum; i++) {
        numCol *= context->GetInputShape(INPUT_GAMMA_INDEX)->GetStorageShape().GetDim(i);
    }
    tiling.set_numFirstDim(numRow);
    tiling.set_numLastDim(numCol);

    auto firstdimPerCore = CEIL_DIV(tiling.get_numFirstDim(), maxCoreNum);
    uint32_t numCore = CEIL_DIV(tiling.get_numFirstDim(), firstdimPerCore);
    int32_t workspaceSize = 1;
    tiling.set_workspaceSize(workspaceSize);
    tiling.set_numCore(numCore);
    context->SetBlockDim(numCore);
    tiling.set_firstDimPerCore(CEIL_DIV(numRow, numCore));
    int32_t nlFirstdimPerCoreNum = tiling.get_firstDimPerCore();
    tiling.set_firstDimPerCoreTail(numRow - nlFirstdimPerCoreNum * (numCore - 1));
    float tempAve = (numCol == 0) ? 0 : float(1.0 / numCol);
    tiling.set_aveFactor(tempAve);
    uint32_t rowPerTime;
    uint32_t colPerTime;

    auto x1DataType = context->GetInputDesc(0)->GetDataType();
    auto x2DataType = context->GetInputDesc(1)->GetDataType();
    auto betagammaDataType = context->GetInputDesc(2)->GetDataType();
    ge::DataType dataType;
    if (x1DataType != x2DataType) {
        dataType = ge::DT_FLOAT;
    } else {
        dataType = x1DataType;
    }
    auto dataTypeSize = GetSizeByDataType(dataType);
    int bufferNum = 1;

    BIAS_TYPE biasType = BIAS_TYPE::NO_BIAS;
    auto biasShape = context->GetOptionalInputShape(4);
    if (biasShape == nullptr) {
        OP_LOGW("AddLayerNorm", "Bias is null");
        biasType = BIAS_TYPE::NO_BIAS;
    } else {
        OP_LOGW("AddLayerNorm", "Bias is not null");
        int x1Size = numRow * numCol;
        int64_t biasSize = 1;
        for (size_t i = 0; i < biasShape->GetStorageShape().GetDimNum(); i++) {
            biasSize *= biasShape->GetStorageShape().GetDim(i);
        }
        biasType = x1Size == biasSize ? (BIAS_TYPE::PRESENT_BIAS) : (BIAS_TYPE::BROADCAST_BIAS);
    }
    OP_LOGI("AddLayerNorm", "Bias Type is %u", static_cast<uint32_t>(biasType));

    auto tilingType = AddLayerNormTilingImpl(maxUbSize,
        dataType,
        betagammaDataType,
        bufferNum,
        numCol,
        enableAdditionalOutput,
        biasType,
        rowPerTime,
        colPerTime,
        is310P);
    if (rowPerTime > firstdimPerCore) {
        rowPerTime = firstdimPerCore;
    }
    tiling.set_firstDimPerTime(rowPerTime);
    int32_t colMoveCnt = CEIL_DIV(numCol, colPerTime);
    int32_t colTail = (numCol % colPerTime == 0) ? colPerTime : (numCol % colPerTime);
    auto blockElementNum = BLOCK_SIZE / dataTypeSize;
    if (is310P) {
        if (((numCol % blockElementNum) != 0) &&
            (tilingType == TILING_TYPE::NORMAL || tilingType == TILING_TYPE::NORMAL_BIG_N ||
                tilingType == TILING_TYPE::NORMAL_SPECIAL_REDUCE ||
                tilingType == TILING_TYPE::NORMAL_BIG_N_SPECIAL_REDUCE)) {
            tilingType = TILING_TYPE::SINGLE_ROW;
            tiling.set_firstDimPerTime(1);
        }
        if (colTail < blockElementNum) {
            auto colSliceNum = CEIL_DIV(numCol, colPerTime);
            colPerTime = CEIL_DIV(numCol, (colSliceNum * blockElementNum)) * blockElementNum;
            colTail = (numCol % colPerTime == 0) ? colPerTime : (numCol % colPerTime);
        }
    }
    tiling.set_lastDimPerTime(colPerTime);
    tiling.set_colMoveCnt(colMoveCnt);
    tiling.set_colTail(colTail);
    if (tilingType == TILING_TYPE::NORMAL && (numCol < BIG_N_MAX_D)) {
        tilingType = TILING_TYPE::NORMAL_BIG_N;
        if (tiling.get_firstDimPerTime() > MAX_REPEAT_TIMES) {
            tiling.set_firstDimPerTime(MAX_REPEAT_TIMES);
            OP_LOGD("AddLayerNorm", "BIG_N_MAX_D, rowPerTime = %d", MAX_REPEAT_TIMES);
        }
    }
    if (tilingType == TILING_TYPE::NORMAL_SPECIAL_REDUCE && (numCol < BIG_N_MAX_D)) {
        tilingType = TILING_TYPE::NORMAL_BIG_N_SPECIAL_REDUCE;
        if (tiling.get_firstDimPerTime() > MAX_REPEAT_TIMES) {
            tiling.set_firstDimPerTime(MAX_REPEAT_TIMES);
            OP_LOGD("AddLayerNorm", "BIG_N_MAX_D_310, rowPerTime = %d", MAX_REPEAT_TIMES);
        }
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());

    uint64_t tilingKey = 0;
    if (enableAdditionalOutput) {
        tilingKey += TILING_ADDITIONAL_OUTPUT;
    }
    if (tilingType == TILING_TYPE::SLICE) {
        tilingKey += TILING_SLICED;
    } else if (tilingType == TILING_TYPE::SINGLE_ROW) {
        tilingKey += TILING_SINGLE_ROW;
    } else if (tilingType == TILING_TYPE::SINGLE_ROW_EXT) {
        tilingKey += TILING_SINGLE_ROW_EXT;
    } else if (tilingType == TILING_TYPE::NORMAL_BIG_N) {
        tilingKey += TILING_NORMAL_BIG_N;
    } else if (tilingType == TILING_TYPE::SLICE_EXT) {
        tilingKey += TILING_SLICE_EXT;
    } else if (tilingType == TILING_TYPE::NORMAL_SPECIAL) {
        tilingKey += TILING_NORMAL_SPECIAL;
    } else if (tilingType == TILING_TYPE::NORMAL_SPECIAL_REDUCE) {
        tilingKey += TILING_NORMAL_SPECIAL_REDUCE;
    } else if (tilingType == TILING_TYPE::NORMAL_BIG_N_SPECIAL_REDUCE) {
        tilingKey += TILING_NORMAL_BIG_N_SPECIAL_REDUCE;
    } else if (tilingType == TILING_TYPE::SINGLE_ROW_LESS_TENSOR) {
        tilingKey += TILING_SINGLE_ROW_LESS_TENSOR;
    }

    if (biasType == BIAS_TYPE::BROADCAST_BIAS) {
        tilingKey += TILING_X3_BROADCAST;
    } else if (biasType == BIAS_TYPE::PRESENT_BIAS) {
        tilingKey += TILING_X3_PRESENT;
    }

    context->SetTilingKey(tilingKey);

    OP_LOGD("AddLayerNorm", "eps = %f", eps);
    OP_LOGD("AddLayerNorm", "numRow = %d", numRow);
    OP_LOGD("AddLayerNorm", "numCol = %d", numCol);
    OP_LOGD("AddLayerNorm", "workspaceSize = %d, numCore = %u", workspaceSize, numCore);
    OP_LOGD("AddLayerNorm", "firstDimPerCore = %u", CEIL_DIV(numRow, numCore));
    OP_LOGD("AddLayerNorm", "firstDimPerCoreTail = %d", (numRow - nlFirstdimPerCoreNum * (numCore - 1)));
    OP_LOGD("AddLayerNorm", "tempAve = %f", tempAve);
    OP_LOGD("AddLayerNorm", "rowPerTime = %u", rowPerTime);
    OP_LOGD("AddLayerNorm", "colPerTime = %u", colPerTime);
    OP_LOGD("AddLayerNorm", "colMoveCnt = %d", colMoveCnt);
    OP_LOGD("AddLayerNorm", "colTail = %d", colTail);
    OP_LOGD("AddLayerNorm", "tilingKey = %lu", tilingKey);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_910B;
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_310P;
    }
    currentWorkspace[0] = workspaceSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4AddLayerNorm(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
struct AddLayerNormCompileInfo {};

IMPL_OP_OPTILING(AddLayerNorm)
    .Tiling(Tiling4AddLayerNorm)
    .TilingParse<AddLayerNormCompileInfo>(TilingPrepare4AddLayerNorm);
}  // namespace optiling
