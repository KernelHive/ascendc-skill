/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl_stft.h"
#include "stft.h"
#include "padv3.h"
#include "mul.h"
#include "ones_like.h"
#include "contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/framework_op.h"
#include "platform/platform_info.h"
#include "aclnn_kernels/common/op_error_check.h"
#include <cmath>
#include <mutex>
#include <map>
#include <string>

using namespace op;

static const uint64_t STFT_MIN_INPUT_DIM = 1;
static const uint64_t STFT_MAX_INPUT_DIM = 2;
static const uint64_t STFT_WINDOW_DIM = 1;
static const uint64_t STFT_MIN_OUTPUT_DIM = 2;
static const uint64_t STFT_MAX_OUTPUT_DIM = 4;
static const int64_t PAD_VALUE = 0;
static const std::string PAD_MODE = "constant";
static const float K2PI = 6.2831853071795864769252867665590057683943388f;
static const int QUADRANT_ONE = 1;
static const int QUADRANT_TWO = 2;
static const int QUADRANT_FOUR = 4;
static const int REAL_IMAG_NUM = 2;
static const int DEVICE_MAX_CACHE_NUM = 5;
static const int FP32_DIVIDE_FP16 = 2;
static const int FP16_NUM_PER_BLOCK = 16;
static const int X1_NFFT = 400;
static const int X1_HOP = 160;
static const int X1_ROW_SIZE = 201;
static const int X1_BATCH = 16;
static const int ROW_SIZE_DIVIDE = 3;
static const int ROW_SIZE_DIVIDE_B3 = 5;
static const int SECOND_ROW_SIZE_DIVIDE = 2;
static const int BLOCK_SIZE = 32;
static const int PACKAGE_SIZE = 128;
static const int FP32_BYTES = 4;

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_DOUBLE, DataType::DT_COMPLEX64, DataType::DT_COMPLEX128};

struct PlanCacheKey {
  int64_t row;
  int64_t col;
  int64_t hopLength;
  int64_t winLength;
  bool normalized;
  bool onesided;
  bool returnComplex;
  int32_t deviceId;
};

struct PlanCacheKeyHash {
  std::size_t operator()(const PlanCacheKey& key) const {
    return ((key.row << 32) | (key.col & 0xffffffff)) + key.hopLength + key.winLength + key.normalized + key.onesided +
           key.returnComplex + key.deviceId;
  }
};

struct PlanCacheKeyEqual {
  bool operator()(const PlanCacheKey& lhs, const PlanCacheKey& rhs) const {
    return (lhs.row == rhs.row) && (lhs.col == rhs.col) && (lhs.hopLength == rhs.hopLength) &&
           (lhs.winLength == rhs.winLength) && (lhs.normalized == rhs.normalized) && (lhs.onesided == rhs.onesided) &&
           (lhs.returnComplex == rhs.returnComplex) && (lhs.deviceId == rhs.deviceId);
  }
};

class StftSingleton {
 private:
  std::mutex cacheNumMutex;
  std::mutex planCacheMutex;

  std::map<int32_t, int> deviceCacheNum;
  std::unordered_map<PlanCacheKey, void*, PlanCacheKeyHash, PlanCacheKeyEqual> planCache;

 public:
  static StftSingleton& GetInstance() {
    static StftSingleton instance;
    return instance;
  }

  void addCacheNum(int32_t deviceId) {
    std::lock_guard<std::mutex> lock(cacheNumMutex);
    deviceCacheNum[deviceId]++;
  }

  int findCacheNum(int32_t deviceId) {
    std::lock_guard<std::mutex> lock(cacheNumMutex);
    return deviceCacheNum[deviceId];
  }

  void addPlanCache(int64_t rowSize, int64_t colSize, int64_t hopLength, int64_t winLength, bool normalized,
                    bool onesided, bool returnComplex, int32_t deviceId, void* planDevice) {
    std::lock_guard<std::mutex> lock(planCacheMutex);
    PlanCacheKey key = {rowSize, colSize, hopLength, winLength, normalized, onesided, returnComplex, deviceId};
    auto it = planCache.find(key);
    if (it == planCache.end()) {
      planCache[key] = planDevice;
    }
  }

  void* findPlanCache(int64_t rowSize, int64_t colSize, int64_t hopLength, int64_t winLength, bool normalized,
                      bool onesided, bool returnComplex, int32_t deviceId) {
    std::lock_guard<std::mutex> lock(planCacheMutex);
    PlanCacheKey key = {rowSize, colSize, hopLength, winLength, normalized, onesided, returnComplex, deviceId};
    auto it = planCache.find(key);
    if (it != planCache.end()) {
      return it->second;
    }
    return nullptr;
  }
};

static int64_t nFftToAlign(const aclTensor* self, int64_t nfft, int alignBytes) {
  int64_t nFftAlign = 0;
  switch (self->GetDataType()) {
    case DataType::DT_FLOAT: {
      int alignNum = alignBytes / FP32_BYTES;
      nFftAlign = (nfft + alignNum - 1) / alignNum * alignNum;
      break;
    }
    default:
      break;
  }

  return nFftAlign;
}

static int NfftAlignBytes(int64_t nfft, int64_t hopLength, bool normalized, bool onesided, bool returnComplex) {
  if (nfft == X1_NFFT && hopLength == X1_HOP && normalized == false && onesided == true && returnComplex == false) {
    return BLOCK_SIZE;
  }
  return PACKAGE_SIZE;
}

static float Mul2Pi(int m, int n) {
  if (n == 0) {
    return -1;
  }
  return ((K2PI * (m)) / (n));
}

static void CalcRealAndImag(int m, int n, float* out) {
  int m0 = m;
  int n0 = n;
  float* out0 = out;
  float theta, c, s, t;
  unsigned int octant = 0;
  int size = n0;

  m0 = m0 % n0;
  n0 += n0;
  n0 += n0;
  m0 += m0;
  m0 += m0;

  if (m0 < 0) {
    m0 += n0;
  }
  if (m0 > n0 - m0) {
    m0 = n0 - m0;
    octant |= QUADRANT_FOUR;
  }
  if (m0 > size) {
    m0 = m0 - size;
    octant |= QUADRANT_TWO;
  }
  if (m0 > size - m0) {
    m0 = size - m0;
    octant |= QUADRANT_ONE;
  }

  theta = Mul2Pi(m0, n0);
  c = cos(theta);
  s = sin(theta);

  if ((octant & QUADRANT_ONE) != 0) {
    t = c;
    c = s;
    s = t;
  }
  if ((octant & QUADRANT_TWO) != 0) {
    t = c;
    c = -s;
    s = t;
  }
  if ((octant & QUADRANT_FOUR) != 0) {
    s = -s;
  }
  out0[0] = c;
  out0[1] = s;
}

static bool HasEmptyTensor(const aclTensor* self) {
  // 检查张量是否存在空维
  if (self->IsEmpty()) {
    return true;
  }

  return false;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* window, const aclTensor* out) {
  // 检查self, window, out的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, return false);
  if (window != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(window, ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(self, window, return false);
  }
  OP_CHECK_DTYPE_NOT_SUPPORT(out, ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, return false);

  return true;
}

static bool CheckFormat(const aclTensor* self) {
  // self格式是ND
  if (self->GetStorageFormat() != Format::FORMAT_ND) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input format only support ND");
    return false;
  }
  return true;
}

static op::Shape GetOutputShape(const aclTensor* self, bool onesided, bool returnComplex, int64_t hopLength,
                                int64_t nFft) {
  op::Shape selfShape = self->GetViewShape();
  auto dimNum = selfShape.GetDimNum();
  int64_t batch = dimNum == STFT_MAX_INPUT_DIM ? selfShape.GetDim(0) : 0;
  int64_t len = dimNum == STFT_MAX_INPUT_DIM ? selfShape.GetDim(1) : selfShape.GetDim(0);
  int64_t frames = (len - nFft) / hopLength + 1;
  int64_t n = onesided == true ? nFft / REAL_IMAG_NUM + 1 : nFft;

  op::Shape outShape;
  op::Shape outShapeComplexWithBatch = {batch, n, frames};
  op::Shape outShapeComplex = {n, frames};
  op::Shape outShapeRealWithBatch = {batch, n, frames, REAL_IMAG_NUM};
  op::Shape outShapeReal = {n, frames, REAL_IMAG_NUM};

  if (returnComplex) {
    outShape = batch > 0 ? outShapeComplexWithBatch : outShapeComplex;
  } else {
    outShape = batch > 0 ? outShapeRealWithBatch : outShapeReal;
  }
  return outShape;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out, const aclTensor* window, int64_t hopLength,
                       int64_t winLength, int64_t nFft, bool onesided, bool returnComplex) {
  // input dim: 1~2
  OP_CHECK_MIN_DIM(self, STFT_MIN_INPUT_DIM, return false);
  OP_CHECK_MAX_DIM(self, STFT_MAX_INPUT_DIM, return false);

  // output dim: 2~4
  OP_CHECK_MIN_DIM(out, STFT_MIN_OUTPUT_DIM, return false);
  OP_CHECK_MAX_DIM(out, STFT_MAX_OUTPUT_DIM, return false);

  op::Shape selfShape = self->GetViewShape();
  auto dimNum = selfShape.GetDimNum();
  int64_t len = dimNum == STFT_MAX_INPUT_DIM ? selfShape.GetDim(1) : selfShape.GetDim(0);
  if (nFft <= 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect nFft > 0");
    return false;
  }
  if (len < nFft) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect input length >= nFft");
    return false;
  }
  if (hopLength <= 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect hopLength > 0");
    return false;
  }
  if (winLength <= 0 || winLength > nFft) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect 0 < winLength <= nFft");
    return false;
  }
  bool isInputComplex = false;
  if (self->GetDataType() == DataType::DT_COMPLEX64 || self->GetDataType() == DataType::DT_COMPLEX128) {
    isInputComplex = true;
  }
  if (window) {
    OP_CHECK_MIN_DIM(window, STFT_WINDOW_DIM, return false);
    OP_CHECK_MAX_DIM(window, STFT_WINDOW_DIM, return false);
    // winLength不等于nfft时需要和window的shape相同
    if (winLength != nFft && window->GetViewShape().GetDim(0) != winLength) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect winLength and window size should be equal");
      return false;
    }
    if (window->GetDataType() == DataType::DT_COMPLEX64 || window->GetDataType() == DataType::DT_COMPLEX128) {
      isInputComplex = true;
    }
  }
  // if input is complex, onesided can't be true
  if (isInputComplex && onesided) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when input is complex, onesided can't be true");
      return false;
  }
  op::Shape outShape = GetOutputShape(self, onesided, returnComplex, hopLength, nFft);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, outShape, return false);

  return true;
}

static bool CheckPlatform() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return true;
  } else {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "STFT is not supported on this platform");
    return false;
  }
}


static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out, const aclTensor* window, int64_t hopLength,
                               int64_t winLength, int64_t nFft, bool onesided, bool returnComplex) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, window, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查格式是否支持
  CHECK_RET(CheckFormat(self), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, out, window, hopLength, winLength, nFft, onesided, returnComplex),
            ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static const aclTensor* GeneratePadWindow(const aclTensor* self, const aclTensor* window, int64_t winLength,
                                          int64_t nFft, int nfftAlignBytes, aclOpExecutor* executor) {
  int64_t left = (nFft - winLength) / 2;

  // nFft按照block对齐，即nFft -> nFft_align
  int64_t nFftAlign = nFftToAlign(self, nFft, nfftAlignBytes);
  int64_t right = nFftAlign - winLength - left;
  if (window == nullptr) {
    auto assist = executor->AllocHostTensor({winLength}, DataType::DT_FLOAT);
    window = l0op::OnesLike(assist, executor);
  }
  // 生成填充tensor
  size_t dims = 2;
  std::vector<int64_t> padVec = {left, right};
  auto padArray = executor->AllocIntArray(padVec.data(), dims);
  if (padArray == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc padVec failed");
    return nullptr;
  }
  auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);

  const aclTensor* valueTensor = executor->ConvertToTensor(executor->AllocScalar(PAD_VALUE), window->GetDataType());
  if (valueTensor == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try convert PAD_VALUE pad tensor failed.");
    return nullptr;
  }
  return l0op::PadV3(window, padTensor, valueTensor, PAD_MODE, true, executor);
}

static const aclTensor* GenerateDftMatrix(const aclTensor* self, int64_t rowSize, int64_t colSize, int64_t hopLength,
                                          int64_t winLength, bool normalized, bool onesided, bool returnComplex,
                                          int nfftAlignBytes, aclOpExecutor* executor) {
  // colSize按照block对齐，即(K, nFft) -> (K, nFft_align)
  int64_t colSizeAlign = nFftToAlign(self, colSize, nfftAlignBytes);
  auto deviceId = GetCurrentPlatformInfo().GetDeviceId();
  void* planDevice = StftSingleton::GetInstance().findPlanCache(rowSize, colSize, hopLength, winLength, normalized,
                                                                onesided, returnComplex, deviceId);

  // 命中plan cache
  if (planDevice != nullptr) {
    auto dft = executor->AllocTensor({REAL_IMAG_NUM, rowSize, colSizeAlign}, op::DataType::DT_FLOAT);
    dft->SetFromWorkspace(false);
    dft->SetStorageAddr(planDevice);
    executor->AbandonCache();
    return dft;
  }

  // 未命中plan cache
  auto dftMatrix = executor->AllocHostTensor({2, rowSize, colSizeAlign}, op::DataType::DT_FLOAT);
  float* addrReal = static_cast<float*>(dftMatrix->GetStorageAddr());
  float* addrImag = static_cast<float*>(dftMatrix->GetStorageAddr()) + rowSize * colSizeAlign;
  float out[2];

  // 实部及虚部交错
  addrImag = static_cast<float*>(dftMatrix->GetStorageAddr()) + colSizeAlign;
  for (int i = 0; i < rowSize; i++) {
    if (i > 0) {
      addrReal += colSizeAlign;
      addrImag += colSizeAlign;
    }
    for (int j = 0; j < colSizeAlign; j++) {
      if (j < colSize) {
        CalcRealAndImag(-1 * i * j, colSize, out);
        *addrReal = out[0];
        *addrImag = out[1];
      } else {
        *addrReal = 0;
        *addrImag = 0;
      }
      addrReal++;
      addrImag++;
    }
  }

  const aclTensor* deviceTensor = nullptr;
  auto deviceIdCacheNum = StftSingleton::GetInstance().findCacheNum(deviceId);
  // 判断当前device上plan cache个数是否达到上限
  if (deviceIdCacheNum < DEVICE_MAX_CACHE_NUM) {
    StftSingleton::GetInstance().addCacheNum(deviceId);
    deviceTensor = op::CopyToNpuSync(dftMatrix, executor);
    CHECK_RET(deviceTensor != nullptr, nullptr);
    StftSingleton::GetInstance().addPlanCache(rowSize, colSize, hopLength, winLength, normalized, onesided,
                                              returnComplex, deviceId, deviceTensor->GetData());
    planDevice = deviceTensor->GetData();
  } else {
    deviceTensor = op::CopyToNpu(dftMatrix, executor);
    CHECK_RET(deviceTensor != nullptr, nullptr);
  }

  return deviceTensor;
}

aclnnStatus aclStftGetWorkspaceSize(const aclTensor* self, const aclTensor* windowOptional, aclTensor* out,
                                    int64_t nFft, int64_t hopLength, int64_t winLength, bool normalized,
                                    bool onesided, bool returnComplex, uint64_t* workspaceSize,
                                    aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclStft, DFX_IN(self, windowOptional, nFft, hopLength, winLength, normalized, onesided, returnComplex),
                 DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, out, windowOptional, hopLength, winLength, nFft, onesided, returnComplex);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (HasEmptyTensor(self)) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("self: nullptr, return");
    return ACLNN_SUCCESS;
  }

  int nfftAlignBytes = NfftAlignBytes(nFft, hopLength, normalized, onesided, returnComplex);

  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (!l0op::IsStftAiCoreSupported(selfContiguous, windowOptional, nFft, hopLength, winLength, normalized, onesided,
                                   returnComplex)) {
    // aicpu
    OP_LOGD("Stft: aicpu");
    auto stftResult = l0op::Stft(selfContiguous, nullptr, windowOptional, nFft, hopLength, winLength, normalized,
                                 onesided, returnComplex, uniqueExecutor.get());
    CHECK_RET(stftResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(stftResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    // window length < nFft, need to pad window
    OP_LOGD("Stft: aicore");
    const aclTensor* windowPad;
    int64_t nFftAlign = nFftToAlign(self, nFft, nfftAlignBytes);
    if (winLength < nFftAlign) {
      windowPad = GeneratePadWindow(self, windowOptional, winLength, nFft, nfftAlignBytes, uniqueExecutor.get());
    } else {
      windowPad = windowOptional;
    }

    // 生成辅助矩阵W：w_real（K，N）+ w_imag（K，N）
    const int64_t K = onesided ? (nFft / 2) + 1 : nFft;
    const int64_t N = nFft;
    const aclTensor* dftMatrix = GenerateDftMatrix(self, K, N, hopLength, winLength, normalized, onesided,
                                                   returnComplex, nfftAlignBytes, uniqueExecutor.get());

    const aclTensor* stftResult;
    if (nFft == X1_NFFT && hopLength == X1_HOP && normalized == false && onesided == true && returnComplex == false) {
      // mul(dftMatrix, windowPad)
      const aclTensor* w = windowPad == nullptr ? dftMatrix : l0op::Mul(dftMatrix, windowPad, uniqueExecutor.get());
      // stft
      stftResult = l0op::Stft(selfContiguous, w, nullptr, nFft, hopLength, winLength, normalized, onesided,
                              returnComplex, uniqueExecutor.get());
    } else {
      // stft
      stftResult = l0op::Stft(selfContiguous, dftMatrix, windowPad, nFft, hopLength, winLength, normalized, onesided,
                              returnComplex, uniqueExecutor.get());
    }
    CHECK_RET(stftResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(stftResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclStft(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclStft);

  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
