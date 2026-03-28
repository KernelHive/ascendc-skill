/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_batch_matmul.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "op_api_def.h"
#include "batch_matmul.h"
#include "fill.h"
#include "mul.h"
#include "aclnn_kernels/transdata.h"
#include "cube_util.h"
#include "matmul_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

/* BatchMatMul 算子的完整计算流程如下:
                self            mat2
                 |               |
   reshape/contiguous        reshape/contiguous
                 |               |
                cast            cast
                 |               |
              transdata      transdata
                  \              /
                  batchmatmul_op_V2
                          |
                      transdata
                          |
                        cast
                          |
                       output
*/
static const int SHAPE_LIMIT = 3;
static const int FIRST_DIM = 0;
static const int PENULTIMATE_DIM = 2;
static const int LAST_DIM = 1;
static const uint64_t NUM_TWO = 2UL;
static const int64_t BLOCK_BYTE_SIZE = 32L;
static const uint64_t BLOCK_CUBE = 16UL;
static const uint64_t L1_CACHE_BYTE_SIZE = 512UL * 1024UL;
static const int64_t CACHELINE = 512L;
static const uint64_t BASIC_ALIGN_16 = 16UL;
static const uint64_t KB_SIZE = 1024UL;
static const uint64_t MB_SIZE = 1024UL * 1024UL;
static const uint64_t MULTI_BATCH_THRES = 4UL;
static const std::vector<uint64_t> SUPPORT_ND2NZ_GM2L0_WITHOUT32B = { 64, 96, 128, 160, 192, 224, 256, 384 };
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = { op::DataType::DT_FLOAT,
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16 };
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WITHOUT_BF16 = { op::DataType::DT_FLOAT,
    op::DataType::DT_FLOAT16 };
static const std::initializer_list<op::DataType> DTYPE_LIST_HALF = { op::DataType::DT_FLOAT16, op::DataType::DT_BF16 };

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *mat2, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(mat2, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *mat2, const aclTensor *bias, const aclTensor *out,
    int8_t cubeMathType)
{
    bool bf16flag = CheckSocVersionIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto dtypeList = bf16flag ? DTYPE_SUPPORT_LIST : DTYPE_SUPPORT_LIST_WITHOUT_BF16;
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mat2, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeList, return false);
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, dtypeList, return false);
    }
    if (!bf16flag && (self->GetDataType() == op::DataType::DT_BF16 || mat2->GetDataType() == op::DataType::DT_BF16 ||
        out->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Bfloat16 is unsupported by the current SOC version [%s], now self is %s, mat2 is %s, out is %s",
            op::ToString(socVersion).GetString(), op::ToString(self->GetDataType()).GetString(),
            op::ToString(mat2->GetDataType()).GetString(), op::ToString(out->GetDataType()).GetString());
        return false;
    }
    if (cubeMathType == KEEP_DTYPE && out->GetDataType() == op::DataType::DT_FLOAT16 &&
        self->GetDataType() == op::DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input tensor's dtype[DT_FLOAT] should be same with output's dtype[DT_FLOAT16].");
        return false;
    }
    if (cubeMathType == KEEP_DTYPE &&
        std::find(DTYPE_LIST_HALF.begin(), DTYPE_LIST_HALF.end(), self->GetDataType()) != DTYPE_LIST_HALF.end() &&
        std::find(DTYPE_LIST_HALF.begin(), DTYPE_LIST_HALF.end(), mat2->GetDataType()) != DTYPE_LIST_HALF.end()) {
        OP_CHECK_DTYPE_NOT_MATCH(self, mat2->GetDataType(), return false);
    }

    bool dtypeMatch = self->GetDataType() == mat2->GetDataType();
    if (!dtypeMatch) {
        OP_LOGW("Self's dtype [%s] and mat2's dtype [%s] are not equal. Promotion of Data Type will be applied",
            op::ToString(self->GetDataType()).GetString(), op::ToString(mat2->GetDataType()).GetString());
    }
    return true;
}

static bool CheckShape(const aclTensor *selfTensor, const aclTensor *otherTensor, const aclTensor *outTensor)
{
    // 限制DIM必须为3D否则报错
    OP_CHECK_WRONG_DIMENSION(selfTensor, SHAPE_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(otherTensor, SHAPE_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(outTensor, SHAPE_LIMIT, return false);
    auto selfDimNum = selfTensor->GetViewShape().GetDimNum();
    auto otherDimNum = otherTensor->GetViewShape().GetDimNum();
    auto outDimNum = outTensor->GetViewShape().GetDimNum();
    const op::Shape self = selfTensor->GetViewShape();
    const op::Shape other = otherTensor->GetViewShape();
    const op::Shape out = outTensor->GetViewShape();
    // selfDimNum - 1 means self's last dim, and otherDimNum - 2 means mat2's penultimate dim
    if (selfDimNum < 2 || otherDimNum < 2 || outDimNum < 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "shapedim of self, other or out must > 2, actual selfshapeDim [%ld], otherDimNum [%ld] , outDimNum [%ld].",
            selfDimNum, otherDimNum, outDimNum);
        return false;
    }
    if (self[selfDimNum - 1] != other[otherDimNum - PENULTIMATE_DIM]) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "self's last dim and mat2's penultimate dim shoule be same, self [%ld], mat2 [%ld].",
            self[selfDimNum - LAST_DIM], other[otherDimNum - PENULTIMATE_DIM]);
        return false;
    }
    if (self[FIRST_DIM] != other[FIRST_DIM] && self[FIRST_DIM] != 1 && other[FIRST_DIM] != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "self's first dim and mat2's first dim shoule be same, or x1 or x2 equal 1.Now self [%ld], mat2 [%ld].",
            self[FIRST_DIM], other[FIRST_DIM]);
        return false;
    }
    auto firstDim = self[FIRST_DIM] >= other[FIRST_DIM] ? self[FIRST_DIM] : other[FIRST_DIM];
    if (out[outDimNum - PENULTIMATE_DIM] != self[selfDimNum - PENULTIMATE_DIM] ||
        out[outDimNum - LAST_DIM] != other[otherDimNum - LAST_DIM] || out[FIRST_DIM] != firstDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "output's shape is not match input, out_m[%ld] must be same with self_m[%ld], "
            "out_n[%ld] must be same with other_n[%ld], out_batch[%ld] must be same with input_batch[%ld].",
            out[outDimNum - PENULTIMATE_DIM], self[selfDimNum - PENULTIMATE_DIM], out[outDimNum - LAST_DIM],
            other[otherDimNum - LAST_DIM], out[FIRST_DIM], firstDim);
        return false;
    }
    return true;
}

static inline bool CheckMathType(const aclTensor *self, const aclTensor *mat2, int8_t cubeMathType)
{
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = selfFloat || mat2Float ? DataType::DT_FLOAT : DataType::DT_FLOAT16;
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

static const aclTensor *ProcessEmptyTensor(const aclTensor *self, const aclTensor *mat2, aclOpExecutor *executor)
{
    // 获取shape信息
    op::Shape bmmEmptyShape = { (self->GetViewShape())[0], (self->GetViewShape())[1], (mat2->GetViewShape())[2] };
    auto output = executor->AllocTensor(bmmEmptyShape, self->GetDataType());
    if (output->IsEmpty()) {
        OP_LOGI("Returning an empty tensor without actually doing calculation");
        return output;
    }
    FVector<int64_t> fillShape = GetShape(output);
    const aclTensor *dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
    aclIntArray *shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
    const aclScalar *valueScalar = executor->AllocScalar(0);
    const aclTensor *valueTensor = executor->ConvertToTensor(valueScalar, self->GetDataType());
    auto fillTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
    return fillTensor;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *mat2, const aclTensor *out, int8_t cubeMathType)
{
    // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, mat2, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, mat2, nullptr, out, cubeMathType), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和mat2的shape是否符合要求
    CHECK_RET(CheckShape(self, mat2, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查cubeMathType
    CHECK_RET(CheckMathType(self, mat2, cubeMathType), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus SetBatchMatMulOpSupportInfo(const aclTensor *self, const aclTensor *mat2, MmOpInfo &matmulOpInfo,
    int8_t cubeMathType)
{
    // 判断传入L0接口，用于计算的Dtype
    SetMmSupportDType(self, mat2, matmulOpInfo, cubeMathType);

    // 1971场景 ACLNN中BMM全部走ND格式，1980场景进入函数路由
    if (CheckSocVersionIsSupportBf16()) {
        matmulOpInfo.support_info.output_format = Format::FORMAT_ND;
        matmulOpInfo.support_info.self_format = Format::FORMAT_ND;
        matmulOpInfo.support_info.mat2_format = Format::FORMAT_ND;
    } else {
        SetMmSupportFormat(self, mat2, matmulOpInfo);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus GetBatchMatmulOpInfo(const aclTensor *self, const aclTensor *mat2, const aclTensor *out,
    MmOpInfo &matmulOpInfo, int8_t cubeMathType)
{
    matmulOpInfo.ori_info.self_dtype = self->GetDataType();
    matmulOpInfo.ori_info.self_format = GetPrimaryFormat(self->GetStorageFormat());
    matmulOpInfo.ori_info.mat2_dtype = mat2->GetDataType();
    matmulOpInfo.ori_info.mat2_format = GetPrimaryFormat(mat2->GetStorageFormat());
    matmulOpInfo.ori_info.output_dtype = out->GetDataType();
    matmulOpInfo.ori_info.output_format = GetPrimaryFormat(out->GetStorageFormat());
    matmulOpInfo.support_info = matmulOpInfo.ori_info;

    SetBatchMatMulOpSupportInfo(self, mat2, matmulOpInfo, cubeMathType);
    bool inputFp32Flag = matmulOpInfo.support_info.self_dtype == DataType::DT_FLOAT &&
        matmulOpInfo.support_info.mat2_dtype == DataType::DT_FLOAT;
    // 如果允许降精度处理， 则开启HF32模式（0x40），否则采用默认模式; 后续此字段配置需要按照字段表进行配置
    matmulOpInfo.opImplModeEnum =
        inputFp32Flag && ((cubeMathType == ALLOW_FP32_DOWN_PRECISION) || (cubeMathType == USE_HF32)) ? 0x40 : 0x1;
    matmulOpInfo.enableHf32 =
        inputFp32Flag && ((cubeMathType == ALLOW_FP32_DOWN_PRECISION) || (cubeMathType == USE_HF32));
    OP_LOGD("opImplModeEnum=%ld, enableHf32=%d, cubeMathType=%d", matmulOpInfo.opImplModeEnum, matmulOpInfo.enableHf32,
        cubeMathType);

    GetMmInfo(matmulOpInfo);
    return ACLNN_SUCCESS;
}

static const std::initializer_list<std::initializer_list<int64_t>> ASCEND_C_WHITE_LIST = {
    // batchA0, batchA1, batchA2, batchA3, batch_b0, batch_b1, batch_b2, batch_b3, m, k, n, transposeX1, transposeX2,
    // hasBias, DtypeFlag
    { 1, 1, 1, 4800, 1, 1, 1, 4800, 1, 128, 8, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 1, 40, 256, 0, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 1, 300, 256, 0, 0, 0, 0 },
    { 1, 1, 7000, 4, 1, 1, 7000, 4, 1, 100, 32, 0, 1, 0, 0 },
    // for test(fp32)
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 1024, 300, 128, 0, 0, 0, 0 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 1024, 300, 128, 0, 0, 1, 0 },
    // for test(fp16)
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 1024, 300, 128, 0, 0, 0, 1 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 1024, 300, 128, 0, 0, 1, 1 },
    // fp32 whitelist
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 1, 128, 8, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 4, 256, 40, 0, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 1, 256, 40, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 1, 128, 300, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 4, 8, 128, 0, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 4, 40, 256, 0, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 4, 300, 128, 0, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 4, 300, 256, 0, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 1, 4, 976, 0, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 16, 1, 976, 1, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 1, 976, 16, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 4, 1, 976, 1, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 1, 976, 4, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 8, 4, 128, 1, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 4, 128, 8, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 4, 256, 40, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 300, 4, 128, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 4, 128, 300, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 300, 4, 256, 1, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 256, 4, 40, 1, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 128, 4, 300, 1, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 256, 4, 300, 1, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 300, 1, 128, 1, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 300, 1, 256, 1, 0, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 4, 8, 128, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 4, 40, 256, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 4, 300, 128, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 4, 300, 256, 0, 1, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 1, 128, 8, 0, 1, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 1, 256, 40, 0, 1, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 1, 128, 300, 0, 1, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 4, 40, 256, 0, 0, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 4, 300, 128, 0, 0, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 4, 300, 256, 0, 0, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 1, 8, 128, 0, 0, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 1, 40, 256, 0, 0, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 1, 300, 128, 0, 0, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 1, 300, 256, 0, 0, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 1, 4, 976, 0, 0, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 1, 976, 16, 0, 1, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 1, 976, 4, 0, 1, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 4, 128, 8, 0, 1, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 8, 4, 128, 1, 0, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 4, 300, 256, 0, 1, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 300, 4, 128, 1, 0, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 300, 4, 256, 1, 0, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 4, 256, 40, 0, 1, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 4, 300, 128, 0, 1, 0, 0 },
    // 生成式网络case，多batch模板，当前不开放
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 211, 211, 32, 0, 0, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 211, 211, 32, 0, 1, 0, 0 },
    // jd百亿补贴，shape过小且左右矩阵差距大，无法泛化放开
    { 1, 1, 1, 128, 1, 1, 1, 128, 1523, 4, 1, 0, 1, 0, 0 },
    // hf32 whitelist
    { 1, 1, 1, 7000, 1, 1, 1, 7000, 1, 976, 4, 0, 1, 0, 0 },
    { 1, 1, 1, 1024, 1, 1, 1, 1024, 4, 40, 256, 0, 1, 0, 0 },
    { 1, 1, 1, 2400, 1, 1, 1, 2400, 1, 300, 128, 0, 0, 0, 0 },
    { 1, 1, 1, 7000, 1, 1, 1, 7000, 1, 10, 128, 0, 0, 0, 0 },
    { 1, 1, 1, 7000, 1, 1, 1, 7000, 1, 128, 10, 0, 1, 0, 0 },
    // swing transformer
    { 1, 1, 16, 24, 1, 1, 16, 24, 576, 32, 576, 0, 0, 0, 0 },
    { 1, 1, 256, 6, 1, 1, 256, 6, 576, 32, 576, 0, 0, 0, 0 },
    { 1, 1, 64, 12, 1, 1, 64, 12, 576, 32, 576, 0, 0, 0, 0 },
    // for MOE
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 83, 6, 6, 0, 0, 0, 1 },
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 6, 83, 6, 1, 0, 0, 1 },
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 83, 6, 6, 0, 1, 0, 1 },
    { 1, 1, 1, 1500, 1, 1, 1, 1500, 1, 512, 128, 0, 0, 0, 0 },
    { 1, 1, 1, 1500, 1, 1, 1, 1500, 1, 128, 512, 0, 1, 0, 0 },
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 1, 16, 30, 0, 1, 0, 1 },
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 1, 30, 16, 0, 0, 0, 1 },
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 1, 60, 16, 0, 0, 0, 1 },
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 1, 16, 60, 0, 1, 0, 1 },
    { 1, 1, 1, 2, 1, 1, 1, 2, 9016, 8192, 9016, 0, 1, 0, 1 },
    { 1, 1, 1, 2, 1, 1, 1, 2, 9016, 8192, 8192, 0, 0, 0, 1 },
    { 1, 1, 1, 2, 1, 1, 1, 2, 8192, 9016, 8192, 1, 0, 0, 1 },
    { 1, 1, 1, 2, 1, 1, 1, 2, 9016, 8192, 8192, 0, 1, 0, 1 }
};

static const std::initializer_list<std::initializer_list<int64_t>> ASCEND_C_WHITE_B4_LIST = {
    // batchA0, batchA1, batchA2, batchA3, batch_b0, batch_b1, batch_b2, batch_b3, m, k, n, transposeX1, transposeX2,
    // hasBias, DtypeFlag
    // for MOE
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 83, 6, 6, 0, 0, 0, 1 },
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 6, 83, 6, 1, 0, 0, 1 },
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 83, 6, 6, 0, 1, 0, 1 },
    { 1, 1, 1, 1500, 1, 1, 1, 1500, 1, 512, 128, 0, 0, 0, 0 },
    { 1, 1, 1, 1500, 1, 1, 1, 1500, 1, 128, 512, 0, 1, 0, 0 },
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 1, 16, 30, 0, 1, 0, 1 },
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 1, 30, 16, 0, 0, 0, 1 },
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 1, 60, 16, 0, 0, 0, 1 },
    { 1, 1, 1, 2048, 1, 1, 1, 2048, 1, 16, 60, 0, 1, 0, 1 },
    { 1, 1, 1, 2, 1, 1, 1, 2, 9016, 8192, 9016, 0, 1, 0, 1 },
    { 1, 1, 1, 2, 1, 1, 1, 2, 9016, 8192, 8192, 0, 0, 0, 1 },
    { 1, 1, 1, 2, 1, 1, 1, 2, 9016, 8192, 8192, 0, 1, 0, 1 }
};

static inline FVector<int64_t> GetBatchDim(const aclTensor *inputTensor)
{
    size_t tensorDim = inputTensor->GetViewShape().GetDimNum();
    // When dimension > 2 , the case pattern is [B, M, K]
    int64_t batchA3 = tensorDim > 2 ? inputTensor->GetViewShape().GetDim(tensorDim - 3) : 1;
    // When dimension > 3 , the case pattern is [B1, B2, M, K]
    int64_t batchA2 = tensorDim > 3 ? inputTensor->GetViewShape().GetDim(tensorDim - 4) : 1;
    // When dimension > 4 , the case pattern is [B1, B2, B3, M, K]
    int64_t batchA1 = tensorDim > 4 ? inputTensor->GetViewShape().GetDim(tensorDim - 5) : 1;
    // When dimension > 5 , the case pattern is [B1, B2, B3, B4, M, K]
    int64_t batchA0 = tensorDim > 5 ? inputTensor->GetViewShape().GetDim(tensorDim - 6) : 1;
    FVector<int64_t> batchDim = { batchA0, batchA1, batchA2, batchA3 };

    return batchDim;
}

inline static bool IsOnTheWay(ge::Format matFormat, uint64_t innerSize, uint64_t dtypeSize)
{
    if (matFormat == ge::FORMAT_ND) {
        return (std::find(SUPPORT_ND2NZ_GM2L0_WITHOUT32B.begin(), SUPPORT_ND2NZ_GM2L0_WITHOUT32B.end(),
            innerSize * dtypeSize) != SUPPORT_ND2NZ_GM2L0_WITHOUT32B.end());
    }
    return false;
}

inline static bool Is256BAlign(int64_t base, uint64_t dTypeSize)
{
    return (base * dTypeSize % 256 == 0); // 256: align byte size
}

static uint64_t CeilAlign(uint64_t x, uint64_t align)
{
    if (align == 0) {
        return x;
    }
    const uint64_t ratio = x / align;
    return (x % align == 0) ? x : (ratio + 1UL) * align;
};

static bool CheckHitMultiBatchUnAligned(const MmOpInfo &mmOpInfo, BmmNd2nzInfo &bmmNd2nzInfo)
{
    int64_t innerADim = bmmNd2nzInfo.transX1 ? bmmNd2nzInfo.mDim : bmmNd2nzInfo.kDim;
    int64_t innerBDim = bmmNd2nzInfo.transX2 ? bmmNd2nzInfo.kDim : bmmNd2nzInfo.nDim;
    int64_t outerADim = bmmNd2nzInfo.transX1 ? bmmNd2nzInfo.kDim : bmmNd2nzInfo.mDim;
    int64_t outerBDim = bmmNd2nzInfo.transX2 ? bmmNd2nzInfo.nDim : bmmNd2nzInfo.kDim;
    int64_t dtypeASize = bmmNd2nzInfo.dtypeASize;
    int64_t dtypeBSize = bmmNd2nzInfo.dtypeBSize;
    bool innerAlignA = Is256BAlign(innerADim, dtypeASize);
    bool innerAlignB = Is256BAlign(innerBDim, dtypeBSize);
    bool supportNd2NzOnTheFlyA = IsOnTheWay(mmOpInfo.support_info.self_format, innerADim, dtypeASize);
    bool supportNd2NzOnTheFlyB = IsOnTheWay(mmOpInfo.support_info.mat2_format, innerBDim, dtypeBSize);
    bmmNd2nzInfo.nd2nzA = !innerAlignA && !supportNd2NzOnTheFlyA && outerADim > 4; // 外轴<=4会导致数据量增大减慢搬运
    bmmNd2nzInfo.nd2nzA = bmmNd2nzInfo.nd2nzA && (innerADim > 1) &&
        (innerADim * dtypeASize <= 192 ||                        // 192B为最大奇数内轴长度
        (innerADim * dtypeASize <= 384 && innerADim % 2 == 0) || // 384B为最大偶数内轴长度
        (innerADim * dtypeASize <= CACHELINE && innerADim % 4 == 0));
    bmmNd2nzInfo.nd2nzB = !innerAlignB && !supportNd2NzOnTheFlyB && outerBDim > 4; // 外轴<=4会导致数据量增大减慢搬运
    bmmNd2nzInfo.nd2nzB = bmmNd2nzInfo.nd2nzB && (innerBDim > 1) &&
        (innerBDim * dtypeBSize <= 192 ||                        // 192B为最大奇数内轴长度
        (innerBDim * dtypeBSize <= 384 && innerBDim % 2 == 0) || // 384B为最大偶数内轴长度
        (innerBDim * dtypeBSize <= CACHELINE && innerBDim % 4 == 0));
    return (innerADim != BLOCK_BYTE_SIZE / dtypeASize && innerBDim != BLOCK_BYTE_SIZE / dtypeBSize &&
        (bmmNd2nzInfo.nd2nzA || bmmNd2nzInfo.nd2nzB));
}

static bool CheckHitMultiBatchAL1FullLoad(const BmmNd2nzInfo &bmmNd2nzInfo, const aclTensor *x1, const aclTensor *x2,
    uint64_t batchCNum, int coreNum)
{
    if (!(!bmmNd2nzInfo.transX1 && !bmmNd2nzInfo.transX2) ||
        (x1->GetDataType() != DataType::DT_FLOAT || x2->GetDataType() != DataType::DT_FLOAT) ||
        bmmNd2nzInfo.mDim != 1) {
        return false;
    }
    uint64_t shapeM = CeilAlign(static_cast<uint64_t>(bmmNd2nzInfo.mDim), BLOCK_CUBE);
    uint64_t shapeN = CeilAlign(static_cast<uint64_t>(bmmNd2nzInfo.nDim), BLOCK_CUBE);
    uint64_t shapeK = CeilAlign(static_cast<uint64_t>(bmmNd2nzInfo.kDim), BLOCK_CUBE);
    auto floorDiv = [](uint64_t x, uint64_t y) -> uint64_t { return y == 0 ? x : x / y; };
    uint64_t max_elements = 131008; // L1最多能放131008个float32
    uint64_t bBatch = max_elements / (shapeM * shapeK + shapeK * shapeN);
    if (bBatch != 1UL) {
        return false;
    }
    uint64_t aBatch = (max_elements - (shapeK * shapeN)) / (shapeM * shapeK);
    uint64_t preCoreBatch = floorDiv(batchCNum, coreNum);
    // if preCoreBatch < 2, no need use Multibatch
    aBatch = std::max(std::min(aBatch, preCoreBatch), 1UL);
    return aBatch >= 2UL;
}

static bool CheckHitMultiBatch(const BmmNd2nzInfo &bmmNd2nzInfo, uint64_t batchCNum, int coreNum)
{
    uint64_t shapeM = CeilAlign(static_cast<uint64_t>(bmmNd2nzInfo.mDim), BLOCK_CUBE);
    uint64_t shapeN = CeilAlign(static_cast<uint64_t>(bmmNd2nzInfo.nDim), BLOCK_CUBE);
    uint64_t shapeK = CeilAlign(static_cast<uint64_t>(bmmNd2nzInfo.kDim), BLOCK_CUBE);
    auto floorDiv = [](uint64_t x, uint64_t y) -> uint64_t { return y == 0 ? x : x / y; };

    // aclnn目前不支持bias，开放后需要加上biasSize
    uint64_t l1Size = GetL1Size(GetCurrentPlatformInfo().GetSocLongVersion());
    uint64_t iterBatch = floorDiv(l1Size * KB_SIZE, ((shapeM * shapeK + shapeK * shapeN) * bmmNd2nzInfo.dtypeASize));
    if (iterBatch > (static_cast<uint64_t>(UINT32_MAX) / BASIC_ALIGN_16) * BASIC_ALIGN_16) {
        OP_LOGI("iterBatch size is greater than floorAlign(UINT32_MAX, 16) or less zhan 0");
        return false;
    }
    uint64_t preCoreBatch = floorDiv(batchCNum, coreNum);
    // if preCoreBatch < 2, no need use Multibatch
    iterBatch = std::max(std::min(iterBatch, preCoreBatch), 1UL);
    uint64_t loopIndex = floorDiv(batchCNum, iterBatch);
    // 限制搬入大于四次走入多batch模板
    return (iterBatch > 1) && (loopIndex > coreNum * MULTI_BATCH_THRES);
}

inline static bool IsBatchEqual(const FVector<int64_t> &batchDimForX1, const FVector<int64_t> &batchDimForX2)
{
    const size_t dimNumA = batchDimForX1.size();
    const size_t dimNumB = batchDimForX2.size();
    if (dimNumA != dimNumB) {
        return false;
    }
    for (size_t i = 0; i < dimNumA; i++) {
        if (batchDimForX1[i] != batchDimForX2[i]) {
            return false;
        }
    }
    return true;
}

inline static uint64_t GetBatchDimAll(const aclTensor *x)
{
    const FVector<int64_t> batchDims = GetBatchDim(x);
    int64_t result = 1L;
    for (int64_t d : batchDims) {
        result *= d;
    }
    return static_cast<uint64_t>(result);
};

static bool CheckMeetV3ConditionBf16MEqual1(bool transX1, bool transX2, int64_t mDim, int64_t kDim, int64_t nDim)
{
    if (!transX1 && !transX2 && mDim == 1 &&             // mDim大小为1的场景
        (60 <= kDim && kDim <= 300 && kDim % 16 != 0) && // kDim范围在60到300之间并且不是16的倍数
        nDim == 16) {                                    // nDim大小为16的场景
        OP_LOGD("Hit bf16 condition from bmmv2 to bmmv3 when M = 1 and transX2 is false.");
        return true;
    }
    if (!transX1 && transX2 && mDim == 1 &&              // mDim大小为1的场景
        kDim == 16 &&                                    // kDim大小为16的场景
        (60 <= nDim && nDim <= 300 && nDim % 16 != 0)) { // nDim范围在60到300之间并且不是16的倍数
        OP_LOGD("Hit bf16 condition from bmmv2 to bmmv3 when M = 1 and transX2 is true.");
        return true;
    }
    return false;
}

static bool CheckMeetV3ConditionBf16(uint64_t batch, const BmmNd2nzInfo &bmmNd2nzInfo, int64_t mDim, int64_t kDim,
    int64_t nDim)
{
    if (2048 <= batch && batch <= 4096) {                           // 批量大小在2048到4096之间
        if (!bmmNd2nzInfo.transX1 && (64 <= mDim && mDim <= 128) && // mDim范围在64到128之间
            (2 <= kDim && kDim <= 9) &&                             // kDim范围在2到9之间
            (2 <= nDim && nDim <= 9)) {                             // nDim范围在2到9之间
            OP_LOGD("Hit bf16 condition from bmmv2 to bmmv3 when transX1 is false.");
            return true;
        }
        if (bmmNd2nzInfo.transX1 && (2 <= mDim && mDim <= 9) && // mDim范围在2到9之间
            (64 <= kDim && kDim <= 128) &&                      // kDim范围在64到128之间
            (2 <= nDim && nDim <= 9)) {                         // nDim范围在2到9之间
            OP_LOGD("Hit bf16 condition from bmmv2 to bmmv3 when transX1 is true.");
            return true;
        }
        return CheckMeetV3ConditionBf16MEqual1(bmmNd2nzInfo.transX1, bmmNd2nzInfo.transX2, mDim, kDim, nDim);
    }
    return false;
}

static bool CheckMeetV3ConditionFp32(uint64_t batch, bool transX1, int64_t mDim, int64_t kDim, int64_t nDim)
{
    if (1500 <= batch && batch <= 6000) {            // 批量大小在1500到6000之间
        if (!transX1 && mDim == 1 &&                 // mDim大小为1的场景
            (128 <= kDim && kDim <= 1024) &&         // kDim范围在128到1024之间
            (128 <= nDim && nDim <= 1024) &&         // nDim范围在128到1024之间
            ((kDim % 512 == 0 && nDim % 128 == 0) || // kDim为512的倍数并且nDim为128的倍数
            (kDim % 128 == 0 && nDim % 512 == 0))) { // kDim为128的倍数并且nDim为512的倍数
            OP_LOGD("Hit fp32 condition from bmmv2 to bmmv3.");
            return true;
        }
    }
    return false;
}

static bool CheckMeetV3Condition(const aclTensor *x1, const aclTensor *x2, const BmmNd2nzInfo &bmmNd2nzInfo)
{
    if (!IsBatchEqual(GetBatchDim(x1), GetBatchDim(x2))) {
        OP_LOGI("The batchsizes are different.");
        return false;
    }
    if (GetCurrentPlatformInfo().GetSocLongVersion() != "Ascend910B2") {
        OP_LOGI("BMMV3 condition do not support the current SOC.");
        return false;
    }
    uint64_t batch = GetBatchDimAll(x1);
    bool perfBenefits = false;
    if (x1->GetDataType() == DataType::DT_BF16) {
        perfBenefits =
            CheckMeetV3ConditionBf16(batch, bmmNd2nzInfo, bmmNd2nzInfo.mDim, bmmNd2nzInfo.kDim, bmmNd2nzInfo.nDim);
    }
    if (!perfBenefits && x1->GetDataType() == DataType::DT_FLOAT) {
        perfBenefits = CheckMeetV3ConditionFp32(batch, bmmNd2nzInfo.transX1, bmmNd2nzInfo.mDim, bmmNd2nzInfo.kDim,
            bmmNd2nzInfo.nDim);
    }
    if (perfBenefits) {
        OP_LOGI("Hit batch_mat_mul_v3 shape of perf benefit.");
        return true;
    }
    return false;
}

static bool CheckHitV3Shape(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const MmOpInfo &mmOpInfo,
    const bool adjX1, const bool adjX2)
{
    FVector<int64_t> batchDimForX1 = GetBatchDim(x1);
    FVector<int64_t> batchDimForX2 = GetBatchDim(x2);
    BmmNd2nzInfo bmmNd2nzInfo;
    bmmNd2nzInfo.transX1 = adjX1;
    bmmNd2nzInfo.transX2 = adjX2;
    // There are 4 batch_dim in total;Index 1 for batchA1; Index 2 for batchA2; Index 3 for batchA3;
    int64_t batchA0 = batchDimForX1[0];
    int64_t batchA1 = batchDimForX1[1];
    int64_t batchA2 = batchDimForX1[2];
    int64_t batchA3 = batchDimForX1[3];
    // There are 4 batch_dim in total;Index 1 for batchB1; Index 2 for batchB2; Index 3 for batchB3;
    int64_t batchB0 = batchDimForX2[0];
    int64_t batchB1 = batchDimForX2[1];
    int64_t batchB2 = batchDimForX2[2];
    int64_t batchB3 = batchDimForX2[3];
    op::Shape shapeX1 = x1->GetViewShape();
    op::Shape shapeX2 = x2->GetViewShape();
    bmmNd2nzInfo.dtypeASize = ge::GetSizeByDataType(x1->GetDataType());
    bmmNd2nzInfo.dtypeBSize = ge::GetSizeByDataType(x2->GetDataType());

    bmmNd2nzInfo.mDim = adjX1 ? shapeX1[shapeX1.GetDimNum() - 1] : shapeX1[shapeX1.GetDimNum() - NUM_TWO];
    bmmNd2nzInfo.kDim = adjX1 ? shapeX1[shapeX1.GetDimNum() - NUM_TWO] : shapeX1[shapeX1.GetDimNum() - 1];
    bmmNd2nzInfo.nDim = adjX2 ? shapeX2[shapeX2.GetDimNum() - NUM_TWO] : shapeX2[shapeX2.GetDimNum() - 1];
    bool hasBias = bias != nullptr;
    int64_t x1DtypeFlag = (x1->GetDataType() == DataType::DT_FLOAT) ? FP32_FLAG : FP16_BF16_FLAG;

    std::initializer_list<int64_t> checkCase = { batchA0,
        batchA1,
        batchA2,
        batchA3,
        batchB0,
        batchB1,
        batchB2,
        batchB3,
        bmmNd2nzInfo.mDim,
        bmmNd2nzInfo.kDim,
        bmmNd2nzInfo.nDim,
        static_cast<int64_t>(adjX1),
        static_cast<int64_t>(adjX2),
        static_cast<int64_t>(hasBias),
        x1DtypeFlag };
    OP_LOGI("Checking BMM-V3 Case: m = %ld, k = %ld, n = %ld, trans1 = %ld, trans2 = %ld, bias = %ld, dtype-flag = %ld",
        bmmNd2nzInfo.mDim, bmmNd2nzInfo.kDim, bmmNd2nzInfo.nDim, static_cast<int64_t>(adjX1),
        static_cast<int64_t>(adjX2), static_cast<int64_t>(hasBias), x1DtypeFlag);
    auto whiteList =
        (GetCurrentPlatformInfo().GetSocLongVersion() == SOC_B4) ? ASCEND_C_WHITE_B4_LIST : ASCEND_C_WHITE_LIST;
    bool ret = std::any_of(whiteList.begin(), whiteList.end(), [checkCase](std::initializer_list<int64_t> oneCase) {
        return checkCase.size() == oneCase.size() && std::equal(oneCase.begin(), oneCase.end(), checkCase.begin());
    });
    if (ret) {
        OP_LOGI("Hit batch_mat_mul_v3 white list.");
        return ret;
    }
    if (IsBatchEqual(batchDimForX1, batchDimForX2) &&
        CheckHitMultiBatch(bmmNd2nzInfo, GetBatchDimAll(x1), GetCurrentPlatformInfo().GetCubeCoreNum()) &&
        CheckHitMultiBatchUnAligned(mmOpInfo, bmmNd2nzInfo)) {
        OP_LOGI("Hit batch_mat_mul_v3 multi batch unAligned.");
        return true;
    }
    if (IsBatchEqual(batchDimForX1, batchDimForX2) && CheckHitMultiBatchAL1FullLoad(bmmNd2nzInfo, x1, x2,
        GetBatchDimAll(x1), GetCurrentPlatformInfo().GetCubeCoreNum())) {
        OP_LOGI("Hit batch_mat_mul_v3 multi batch AL1 fullLoad.");
        return true;
    }
    return CheckMeetV3Condition(x1, x2, bmmNd2nzInfo);
}

static bool IsL1FullLoadScenario(const aclTensor *x1, const aclTensor *x2)
{
    const uint64_t aBatchDimAll = GetBatchDimAll(x1);
    const uint64_t bBatchDimAll = GetBatchDimAll(x2);
    const bool aNoBatch = aBatchDimAll <= 1;
    const bool bNoBatch = bBatchDimAll <= 1;
    if (aNoBatch == bNoBatch) {
        return false;
    }
    const size_t dtypeSize = op::TypeSize(x1->GetDataType());
    const uint64_t c0 = static_cast<uint64_t>(BLOCK_BYTE_SIZE) / dtypeSize;
    const op::Shape shapeX1 = x1->GetViewShape();
    const size_t dimNumX1 = shapeX1.GetDimNum();
    const uint64_t dimOuterX1 = static_cast<uint64_t>(shapeX1.GetDim(dimNumX1 - PENULTIMATE_DIM));
    const uint64_t dimInnerX1 = static_cast<uint64_t>(shapeX1.GetDim(dimNumX1 - LAST_DIM));
    const op::Shape shapeX2 = x2->GetViewShape();
    const size_t dimNumX2 = shapeX2.GetDimNum();
    const uint64_t dimOuterX2 = static_cast<uint64_t>(shapeX2.GetDim(dimNumX2 - PENULTIMATE_DIM));
    const uint64_t dimInnerX2 = static_cast<uint64_t>(shapeX2.GetDim(dimNumX2 - LAST_DIM));

    auto getMatrixArea = [dtypeSize, c0](uint64_t batch, uint64_t dimOuter, uint64_t dimInner) -> uint64_t {
        auto ceilAlign = [](uint64_t num, uint64_t align) -> uint64_t {
            if (align == 0) {
                return num;
            }
            const uint64_t ratio = num / align;
            return (num % align == 0) ? num : (ratio + 1UL) * align;
        };

        return std::max(batch, 1UL) * ceilAlign(dimOuter, BLOCK_CUBE) * ceilAlign(dimInner, c0) * dtypeSize;
    };
    const uint64_t aSize = getMatrixArea(aBatchDimAll, dimOuterX1, dimInnerX1);
    const uint64_t bSize = getMatrixArea(bBatchDimAll, dimOuterX2, dimInnerX2);
    const uint32_t aicoreNum = GetCurrentPlatformInfo().GetCubeCoreNum();
    // 非驻留tensor的c0在mmv3上没有优化，在MMV2上有优化
    bool isAL1FullLoad = dimInnerX2 != c0 && (aNoBatch && (aSize * NUM_TWO <= L1_CACHE_BYTE_SIZE) &&
        (bSize >= L1_CACHE_BYTE_SIZE * aicoreNum || bBatchDimAll >= 4UL * aicoreNum)); // 设置单核上至少4个batch循环
    bool isBL1FullLoad = dimInnerX1 != c0 && (bNoBatch && (bSize * NUM_TWO <= L1_CACHE_BYTE_SIZE) &&
        (aSize >= L1_CACHE_BYTE_SIZE * aicoreNum || aBatchDimAll >= 4UL * aicoreNum)); // 设置单核上至少4个batch循环
    return isAL1FullLoad || isBL1FullLoad;
}

static bool CheckAscendCScenario(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
    const MmOpInfo &mmOpInfo, const bool adjX1, const bool adjX2)
{
    if ((GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) ||
        mmOpInfo.support_info.self_format != ge::FORMAT_ND || mmOpInfo.support_info.mat2_format != ge::FORMAT_ND) {
        OP_LOGI("Not batch_mat_mul_v3 case for unsupported SOC version or unsupported Format.");
        return false;
    }
    if (IsL1FullLoadScenario(x1, x2)) {
        OP_LOGI("Hit batch_mat_mul_v3 L1fullload case.");
        return true;
    }
    if ((x1->GetDataType() != DataType::DT_FLOAT16 && x1->GetDataType() != DataType::DT_BF16 &&
        x1->GetDataType() != DataType::DT_FLOAT) ||
        (x2->GetDataType() != DataType::DT_FLOAT16 && x2->GetDataType() != DataType::DT_BF16 &&
        x2->GetDataType() != DataType::DT_FLOAT)) {
        OP_LOGI("Not batch_mat_mul_v3 case due to unsupported dtype.");
        return false;
    }
    if (bias != nullptr) {
        OP_LOGI("batch_mat_mul_v3 case does not support bias yet");
        return false;
    }
    return CheckHitV3Shape(x1, x2, bias, mmOpInfo, adjX1, adjX2);
}

static const aclTensor *GetBatchMatmulOp(const aclTensor *selfTransdata, const aclTensor *mat2Transdata,
    const aclTensor *bias, const MmOpInfo &matmulOpInfo, bool adjX1, bool adjX2, const bool offsetX,
    aclOpExecutor *executor)
{
    auto bmmOpOut = selfTransdata;
    if (CheckAscendCScenario(selfTransdata, mat2Transdata, bias, matmulOpInfo, adjX1, adjX2)) {
        OP_LOGI("Hit batch_mat_mul_v3 scenario.");
        bmmOpOut = l0op::BatchMatMulV3Nd(selfTransdata, mat2Transdata, bias, nullptr, adjX1, adjX2, offsetX,
            matmulOpInfo.enableHf32, executor);
        return bmmOpOut;
    }
    // 输入是FP16的场景
    if (matmulOpInfo.support_info.self_dtype == op::DataType::DT_FLOAT16) {
        if (matmulOpInfo.support_info.output_dtype == op::DataType::DT_FLOAT16) {
            // 输入是FP16, 输出是FP16的场景
            if (matmulOpInfo.support_info.self_format == op::Format::FORMAT_ND) {
                bmmOpOut = l0op::BatchMatMulNd(selfTransdata, mat2Transdata, bias, nullptr, adjX1, adjX2, offsetX,
                    matmulOpInfo.opImplModeEnum, executor);
            } else {
                bmmOpOut = l0op::BatchMatMulNzFp162Fp16(selfTransdata, mat2Transdata, bias, nullptr, adjX1, adjX2,
                    offsetX, matmulOpInfo.opImplModeEnum, executor);
            }
        } else {
            // 输入是FP16, 输出是FP32的场景
            if (matmulOpInfo.support_info.self_format == op::Format::FORMAT_ND) {
                bmmOpOut = l0op::BatchMatMulNdFp162Fp32(selfTransdata, mat2Transdata, bias, nullptr, adjX1, adjX2,
                    offsetX, matmulOpInfo.opImplModeEnum, executor);
            } else {
                bmmOpOut = l0op::BatchMatMulNzFp162Fp32(selfTransdata, mat2Transdata, bias, nullptr, adjX1, adjX2,
                    offsetX, matmulOpInfo.opImplModeEnum, executor);
            }
        }
    } else {
        // 输入是FP32/BF16,输出是FP32/BF16的场景
        bmmOpOut = l0op::BatchMatMulNd(selfTransdata, mat2Transdata, bias, nullptr, adjX1, adjX2, offsetX,
            matmulOpInfo.opImplModeEnum, executor);
    }
    return bmmOpOut;
}

const aclTensor *ExecBatchMatmulOpWithBiasAndAttrs(const aclTensor *self, const aclTensor *mat2, const aclTensor *bias,
    const aclTensor *out, bool adjX1, bool adjX2, int8_t cubeMathType, aclOpExecutor *executor)
{
    CHECK_RET(CheckMathType(self, mat2, cubeMathType), nullptr);
    MmOpInfo matmulOpInfo;
    GetBatchMatmulOpInfo(self, mat2, out, matmulOpInfo, cubeMathType);

    auto selfCast = l0op::Cast(self, matmulOpInfo.support_info.self_dtype, executor);
    CHECK_RET(selfCast != nullptr, nullptr);
    auto mat2Cast = l0op::Cast(mat2, matmulOpInfo.support_info.mat2_dtype, executor);
    CHECK_RET(mat2Cast != nullptr, nullptr);

    // k,m,n=1特殊场景
    auto selfReshape = selfCast;
    auto mat2Reshape = mat2Cast;
    bool ifKEqual1 = IfKEqual1(selfCast, matmulOpInfo, adjX1, bias);
    if (ifKEqual1) {
        aclnnStatus kEqual1SelfToMKRes = IfKEqual1Mat2ToKN(selfCast, selfReshape, adjX1, executor);
        CHECK_RET(kEqual1SelfToMKRes == ACLNN_SUCCESS, nullptr);
        aclnnStatus kEqual1Mat2ToKNRes = IfKEqual1Mat2ToKN(mat2Cast, mat2Reshape, adjX2, executor);
        CHECK_RET(kEqual1Mat2ToKNRes == ACLNN_SUCCESS, nullptr);
        OP_LOGI("Hit MatMul or BatchMatmul k=1 scenario, trans matmul to mul to calculate");
    } else {
        aclnnStatus mEqual1SelfToMKRes =
            IfMEqual1SelfToMK(selfCast, selfReshape, matmulOpInfo.support_info.self_format, adjX1, executor);
        CHECK_RET(mEqual1SelfToMKRes == ACLNN_SUCCESS, nullptr);
        aclnnStatus nEqual1Mat2ToNKRes =
            IfNEqual1Mat2ToNK(mat2Cast, mat2Reshape, matmulOpInfo.support_info.mat2_format, adjX2, executor);
        CHECK_RET(nEqual1Mat2ToNKRes == ACLNN_SUCCESS, nullptr);
    }

    auto selfTransdata = l0op::TransData(selfReshape, matmulOpInfo.support_info.self_format, 0, executor);
    CHECK_RET(selfTransdata != nullptr, nullptr);
    auto mat2Transdata = l0op::TransData(mat2Reshape, matmulOpInfo.support_info.mat2_format, 0, executor);
    CHECK_RET(mat2Transdata != nullptr, nullptr);

    const aclTensor *bmmOpOut = nullptr;
    if (ifKEqual1) {
        bmmOpOut = l0op::Mul(selfTransdata, mat2Transdata, executor);
    } else {
        bmmOpOut = GetBatchMatmulOp(selfTransdata, mat2Transdata, bias, matmulOpInfo, adjX1, adjX2, 0, executor);
    }

    CHECK_RET(bmmOpOut != nullptr, nullptr);

    auto transdataOut = l0op::TransData(bmmOpOut, matmulOpInfo.ori_info.output_format, 0, executor);
    CHECK_RET(transdataOut != nullptr, nullptr);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(transdataOut, out->GetDataType(), executor);
    CHECK_RET(castOut != nullptr, nullptr);

    return castOut;
}

const aclTensor *ExecBatchMatmulOp(const aclTensor *self, const aclTensor *mat2, const aclTensor *out, bool adjX1,
    bool adjX2, int8_t cubeMathType, aclOpExecutor *executor)
{
    return ExecBatchMatmulOpWithBiasAndAttrs(self, mat2, nullptr, out, adjX1, adjX2, cubeMathType, executor);
}

const aclTensor *ExecBmmOpWithBias(const aclTensor *self, const aclTensor *mat2, const aclTensor *bias,
    const aclTensor *out, int8_t cubeMathType, aclOpExecutor *executor)
{
    CHECK_RET(CheckNotNull(self, mat2, out), nullptr);
    CHECK_RET(CheckDtypeValid(self, mat2, bias, out, cubeMathType), nullptr);
    CHECK_RET(CheckMathType(self, mat2, cubeMathType), nullptr);
    if (self->IsEmpty() || mat2->IsEmpty()) {
        auto emptyOut = ProcessEmptyTensor(self, mat2, executor);
        CHECK_RET(emptyOut != nullptr, nullptr);
        return emptyOut;
    }

    // reformat，全部转成ND
    auto reformatSelf = l0op::ReFormat(self, op::Format::FORMAT_ND);
    CHECK_RET(reformatSelf != nullptr, nullptr);
    auto reformatMat2 = l0op::ReFormat(mat2, op::Format::FORMAT_ND);
    CHECK_RET(reformatMat2 != nullptr, nullptr);
    auto reformatOut = l0op::ReFormat(out, op::Format::FORMAT_ND);
    CHECK_RET(reformatOut != nullptr, nullptr);

    auto contiguousSelf = reformatSelf;
    auto contiguousMat2 = reformatMat2;

    auto transposeSelf = IsTransposeLastTwoDims(self);
    auto transposeMat2 = IsTransposeLastTwoDims(mat2);

    if (transposeSelf) {
        contiguousSelf = executor->CreateView(self, SwapLastTwoDimValue(self->GetViewShape()), self->GetViewOffset());
    } else {
        contiguousSelf = l0op::Contiguous(self, executor);
    }
    CHECK_RET(contiguousSelf != nullptr, nullptr);

    if (transposeMat2) {
        contiguousMat2 = executor->CreateView(mat2, SwapLastTwoDimValue(mat2->GetViewShape()), mat2->GetViewOffset());
    } else {
        contiguousMat2 = l0op::Contiguous(mat2, executor);
    }
    CHECK_RET(contiguousMat2 != nullptr, nullptr);

    // bias非连续转连续以及转换dtype
    auto contiguousBias = bias;
    if (contiguousBias != nullptr) {
        contiguousBias = ContiguousBias(self, bias, executor);
        CHECK_RET(contiguousBias != nullptr, nullptr);
    }

    auto batchMatmulOut = ExecBatchMatmulOpWithBiasAndAttrs(contiguousSelf, contiguousMat2, contiguousBias, reformatOut,
        transposeSelf, transposeMat2, cubeMathType, executor);

    CHECK_RET(batchMatmulOut != nullptr, nullptr);

    // reformat成原来out的数据格式
    auto reformatBmmOut = l0op::ReFormat(batchMatmulOut, out->GetViewFormat());
    CHECK_RET(reformatBmmOut != nullptr, nullptr);

    return reformatBmmOut;
}

const aclTensor *ExecBmmOp(const aclTensor *self, const aclTensor *mat2, const aclTensor *out, int8_t cubeMathType,
    aclOpExecutor *executor)
{
    return ExecBmmOpWithBias(self, mat2, nullptr, out, cubeMathType, executor);
}

aclnnStatus aclnnBatchMatMulGetWorkspaceSize(const aclTensor *self, const aclTensor *mat2, aclTensor *out,
    int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnBatchMatMul, DFX_IN(self, mat2, cubeMathType), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, mat2, out, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 从最初的接口进入bmm计算
    auto bmmOut = ExecBmmOp(self, mat2, out, cubeMathType, uniqueExecutor.get());
    CHECK_RET(bmmOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (bmmOut->IsEmpty()) {
        // 当输出为空tensor的场景，空tensor处理
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto viewCopyResult = l0op::ViewCopy(bmmOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBatchMatMul);
    // 固定写法，调用框架能力，完成计算
    OP_LOGD("Entering aclnnBatchMatmul");
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
