/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "acl_rfft1d.h"
#include "rfft1d.h"
#include "aclnn_kernels/pad.h"
#include "mul.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "concat.h"
#include "aclnn_kernels/transpose.h"
#include "ones_like.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/framework_op.h"
#include "aclnn_kernels/common/op_error_check.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <complex>
#include <iterator>
#include <valarray>
#include <map>

using namespace op;

static const uint32_t COMPLEX = 2;
static const uint32_t MIN_OUTPUT_DIMS = 2;
static const uint32_t LAST_DIM = 2;
static const uint32_t POWER_OF_TWO = 2;
static const uint32_t NZ_BORDER = 8;
static const uint32_t NZ_BLOCK = 16;
static const uint32_t MAX_FACTORS_LEN = 3;
static const uint32_t DFT_BORDER_VALUE = 4096;
static const uint32_t RFFT_BORDER_VALUE = 262144;
static const uint32_t DEFAULT_MEMORY_SIZE = 100000;
static const uint32_t HASH_KEY_CONSTANT = 1000000;
static const uint32_t FIRST_FACTOR = 2;
static const uint32_t LAST_FACTOR = 64;
static const double INIT_VALUE = 0.;
static const std::string PAD_MODE = "constant";
static const int64_t PAD_VALUE = 0;

enum NORM_VALUES{BACKWARD=1, FORWARD=2, ORTHO=3};

static const std::initializer_list<DataType> NULL_SUPPORT_LIST = {};
static const int DEVICE_MAX_CACHE_NUM = 100;
static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};

class Rfft1DSingleton {
    private:
    std::mutex cacheNumMutex;
    std::mutex planCacheMutex;
  
    std::map<int32_t, int> deviceCacheNum;
    std::map<int64_t, void*> planCache;
    std::map<int64_t, int64_t> tensorLenCache;

    public:
    static Rfft1DSingleton& GetInstance() {
        static Rfft1DSingleton instance;
        return instance;
    }
  
    void AddCacheNum(int32_t deviceId) {
        std::lock_guard<std::mutex> lock(cacheNumMutex);
        deviceCacheNum[deviceId]++;
    }
  
    int FindCacheNum(int32_t deviceId) {
        std::lock_guard<std::mutex> lock(cacheNumMutex);
        return deviceCacheNum[deviceId];
    }
  
    void AddPlanCache(int64_t len, int32_t deviceId, void* planDevice) {
        int64_t key = len + HASH_KEY_CONSTANT * deviceId;
        planCache[key] = planDevice;
    }
  
    void* FindPlanCache(int64_t len, int32_t deviceId) {
        int64_t key = len + HASH_KEY_CONSTANT * deviceId;
        return planCache[key];
    }

    void AddTensorLen(int64_t len, int64_t tensorLen) {
        tensorLenCache[len] = tensorLen;
    }
  
    int64_t FindTensorLen(int64_t len) {
        return tensorLenCache[len];
    }

    bool operator<(const Rfft1DSingleton& other) const {
        return deviceCacheNum < other.deviceCacheNum;
    }
};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return NULL_SUPPORT_LIST;
    }
}

static bool HasEmptyTensor(const aclTensor* self) {
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

static bool CheckDtypeValid(const aclTensor* self) {
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  
    return true;
}

static bool CheckParamsValid(int64_t n, int64_t dim, int64_t norm, int64_t dims) {
    if (!((n > 0 && n <= RFFT_BORDER_VALUE) || n == -1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "'n' should be in [1, 262144] or -1");
        return false;
    }
    if (!((norm == BACKWARD) || (norm == FORWARD) || (norm == ORTHO))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "'norm' should be equal {BACKWARD, FORWARD, ORTHO} via pytorch call or {1, 2, 3} in other cases");
        return false;
    }
    if (!((dim >= -dims) && (dim < dims))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "'dim' out of range (expected to be in range of [%ld, %ld], but got %ld",-dims ,dims - 1, dim);
        return false;
    }
    return true;
}

static bool CheckPlatform() {
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        return true;
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Rfft1D is not supported on this platform");
        return false;
    }
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out, int64_t n, int64_t dim, int64_t norm, int64_t dims) {
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParamsValid(n, dim, norm, dims), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static std::vector<double> Rfft1DDftGen(int64_t fftLength, int64_t norm) {
    std::vector<double> dftNz;
    double normParam = 1.;

    if (norm != BACKWARD) {
        normParam = norm == FORWARD ? normParam / double(fftLength) : normParam / sqrt(double(fftLength));
    }
    
    size_t fftLenPad = fftLength + (NZ_BORDER - fftLength % NZ_BORDER);
    size_t fftLenPadRow = fftLength + (NZ_BLOCK - fftLength % NZ_BLOCK);

    size_t fftOut = fftLength / COMPLEX + 1;
    dftNz.reserve(fftLenPadRow * fftLenPadRow);

    std::vector<double> dftNd(fftLenPad * fftLenPadRow, INIT_VALUE);
    size_t k = 0;
  
    if (fftLength % NZ_BLOCK && fftLength % NZ_BLOCK <= NZ_BORDER)
    {
        for (int i = 0; i < fftLength; ++i) {
            for (size_t j = 0; j < fftOut; ++j) {
                double param = -2. * M_PI * i * j / double(fftLength);
                dftNd[k] = normParam * cos(param);
                ++k;
                dftNd[k] = normParam * sin(param);
                ++k;
            }
        }
        return dftNd;
    }
    else
    {
        for (int i = 0; i < fftLength; ++i) {
            for (size_t j = 0; j < fftOut; ++j) {
                double param = -2. * M_PI * i * j / double(fftLength);
                dftNd[k] = normParam * cos(param);
                ++k;
                dftNd[k] = normParam * sin(param);
                ++k;
            }
            for (size_t j = 0; j + COMPLEX * fftOut < fftLenPad; ++j) {
              dftNd[k] = INIT_VALUE;
              ++k;
            }
        }
        for (size_t n = 0; n < fftLenPad / NZ_BORDER; ++n) {
            for (size_t i = 0; i < fftLenPadRow; ++i) {
                for (size_t blockIdx = 0; blockIdx < NZ_BORDER; ++blockIdx) {
                    dftNz.emplace_back(dftNd[i * fftLenPad + NZ_BORDER * n + blockIdx]);
                }
            }
        }
        return dftNz;
    }
}

void FftRec(std::vector<std::complex<double>>& x, int N) {
    // Check if it is splitted enough
    if (N <= 1) {
        return;
    }

    // Split even and odd
    std::vector<std::complex<double>> odd(N / COMPLEX);
    std::vector<std::complex<double>> even(N / COMPLEX);
    for (size_t i = 0; i < N / COMPLEX; ++i) {
        even[i] = x[i * COMPLEX];
        odd[i] = x[i * COMPLEX + 1];
    }

    // Split on tasks
    FftRec(even, N / COMPLEX);
    FftRec(odd, N / COMPLEX);

    // Calculate DFT
    for (size_t k = 0; k < N / COMPLEX; k++) {
        std::complex<double> t = exp(std::complex<double>(0, -2 * M_PI * k / N)) * odd[k];
        x[k] = even[k] + t;
        x[N / COMPLEX + k] = even[k] - t;
    }
}

void DoubleRI(std::vector<double>& ret, std::vector<std::complex<double>>& betas) {
    uint8_t betasAmount = 4;
    std::vector<double> doubleRI(betas.size() * betasAmount);
    for (size_t k = 0; k < COMPLEX; ++k) {
        for (size_t i = 0; i < betas.size(); ++i) {
            if (k == 0) {
                ret.emplace_back(betas[i].real());
                ret.emplace_back(betas[i].real());
            } else {
                ret.emplace_back(-betas[i].imag());
                ret.emplace_back(betas[i].imag());
            }
        }
    } 
}

static void CalculateBeta(std::complex<double>& b, std::vector<double>& betaTmp,
                          std::vector<double>& betaReverseReal, std::vector<double>& betaReverseImag,
                          std::vector<double>& betaConj, std::vector<double>& betaConjImag,
                          const int64_t len, const uint64_t lenPow2, const int64_t startIndex) {
    for (int64_t i = 0; i < len; ++i) {
        std::complex<double> curBeta = std::pow(b, i * i);

        betaTmp[COMPLEX * i] = curBeta.real();
        betaTmp[COMPLEX * i + 1] = curBeta.real();                       

        betaConj[COMPLEX * i] = curBeta.real();
        betaConj[COMPLEX * i + 1] = curBeta.real();	           
                    
        betaConjImag[COMPLEX * i] = curBeta.imag();
        betaConjImag[COMPLEX * i + 1] = -curBeta.imag();
    }

    std::copy(betaTmp.begin(), betaTmp.end(), betaReverseReal.begin());
    std::reverse(betaReverseReal.begin() + COMPLEX, betaReverseReal.begin() + COMPLEX * len);

    std::copy(betaConjImag.begin(), betaConjImag.end(), betaReverseImag.begin());
    std::reverse(betaReverseImag.begin() + COMPLEX, betaReverseImag.begin() + COMPLEX * len);

    for (uint64_t i = startIndex; i < lenPow2; ++i) {
        betaTmp[COMPLEX * i] = betaReverseReal[COMPLEX + COMPLEX * i - COMPLEX * startIndex + 1];
        betaTmp[COMPLEX * i + 1] = betaReverseReal[COMPLEX + COMPLEX * i - COMPLEX * startIndex + 1];

        betaConj[COMPLEX * i] = betaTmp[COMPLEX * i];
        betaConj[COMPLEX * i + 1] = betaTmp[COMPLEX * i];

        betaConjImag[COMPLEX * i] = betaReverseImag[COMPLEX + COMPLEX * i - COMPLEX * startIndex + 1];
        betaConjImag[COMPLEX * i + 1] = -betaReverseImag[COMPLEX + COMPLEX * i - COMPLEX * startIndex + 1];
    }
}


std::vector<double> GenerateAlphaBeta(const int64_t& len, const uint64_t& lenPow2){
    std::vector<double> ret;
    std::vector<double> ret2;
    std::vector<std::complex<double>> betaComplex(lenPow2);
    double param = - 2 * M_PI / len;
    const double wReal = cos(param), wImag = sin(param);

    std::complex<double> w(wReal, wImag);
    std::complex<double> B = std::pow(w, -0.5);

    { 
        std::vector<double> alpha(len * COMPLEX, 0.0);
        for (int64_t i = 0; i < len; ++i) {
            std::complex<double> curAlpha = 1. / std::pow(B, i * i);
            alpha[COMPLEX * i] = curAlpha.real();
            alpha[COMPLEX * i + 1] = curAlpha.imag();
        }
        ret.insert(ret.end(), std::make_move_iterator(alpha.begin()), std::make_move_iterator(alpha.end()));
    }
    {
        std::vector<double> beta(lenPow2 * COMPLEX, 0.0);
        std::vector<double> betaTmp(lenPow2 * COMPLEX, 0.0);
        std::vector<double> betaReverseReal(lenPow2 * COMPLEX, 0.0);
        std::vector<double> betaReverseImag(lenPow2 * COMPLEX, 0.0);
        std::vector<double> betaReverseTmp(lenPow2 * COMPLEX, 0.0);
        std::vector<double> betaConj(lenPow2 * COMPLEX, 0.0);
        std::vector<double> betaConjImag(lenPow2 * COMPLEX, 0.0);
        
        const uint64_t startIndex = len + (lenPow2 - COMPLEX * len + 1);

        CalculateBeta(B, betaTmp, betaReverseReal, betaReverseImag,
                      betaConj, betaConjImag, len, lenPow2, startIndex);

        ret.insert(ret.end(), std::make_move_iterator(betaConj.begin()), std::make_move_iterator(betaConj.end()));
        ret.insert(ret.end(), std::make_move_iterator(betaConjImag.begin()), std::make_move_iterator(betaConjImag.end()));
        
        for (int64_t i = 0; i < len; ++i) {
            std::complex<double> curBeta = std::pow(B, i * i);
            beta[COMPLEX * i] = curBeta.real();
            beta[COMPLEX * i + 1] = curBeta.imag();
            betaComplex[i] = curBeta;
        }
            
        std::copy(beta.begin(), beta.end(), betaReverseTmp.begin());
        std::reverse(betaReverseTmp.begin() + COMPLEX, betaReverseTmp.begin() + COMPLEX * len);

        for (uint64_t i = startIndex; i < lenPow2; ++i) {
            beta[COMPLEX * i] = betaReverseTmp[COMPLEX + COMPLEX * i - COMPLEX * startIndex + 1];
            beta[COMPLEX * i + 1] = betaReverseTmp[COMPLEX + COMPLEX * i - COMPLEX * startIndex];
            betaComplex[i] = std::complex<double>(beta[COMPLEX * i], beta[COMPLEX * i + 1]);
        }
    }
    {
        FftRec(betaComplex, lenPow2);
        DoubleRI(ret, betaComplex);
    }
    return ret;
}

static const aclTensor* GenerateDftMatrix(int64_t len, int64_t norm, aclOpExecutor* executor) {
    int64_t alignLen = len + (NZ_BORDER - len % NZ_BORDER);
    auto deviceId = GetCurrentPlatformInfo().GetDeviceId();
  
    void* planDevice = Rfft1DSingleton::GetInstance().FindPlanCache(len, deviceId);
    if (planDevice != nullptr) {
        int64_t tensorLen = Rfft1DSingleton::GetInstance().FindTensorLen(len);
        auto dft = executor->AllocTensor({tensorLen}, op::DataType::DT_FLOAT);
        dft->SetFromWorkspace(false);
        dft->SetStorageAddr(planDevice);
        executor->AbandonCache();
        return dft;
    }
  
    auto dft = Rfft1DDftGen(len, norm);
    auto dftMatrix = executor->AllocHostTensor({dft.size()}, op::DataType::DT_FLOAT);
    float* addrStart = static_cast<float*>(dftMatrix->GetStorageAddr());
  
    float mx = 0.0;
    for (size_t i = 0; i < dft.size(); ++i) {
        *(addrStart + i) = float(dft[i]);
    }
  
    const aclTensor* deviceTensor = nullptr;
    auto deviceIdCacheNum = Rfft1DSingleton::GetInstance().FindCacheNum(deviceId);
    Rfft1DSingleton::GetInstance().AddTensorLen(len, dft.size());
    if (deviceIdCacheNum < DEVICE_MAX_CACHE_NUM) {
        Rfft1DSingleton::GetInstance().AddCacheNum(deviceId);
        deviceTensor = op::CopyToNpuSync(dftMatrix, executor);
        CHECK_RET(deviceTensor != nullptr, nullptr);
        Rfft1DSingleton::GetInstance().AddPlanCache(len, deviceId, deviceTensor->GetData());
    } else {
        deviceTensor = op::CopyToNpu(dftMatrix, executor);
        CHECK_RET(deviceTensor != nullptr, nullptr);
    }
  
    return deviceTensor;
}

static void CalculateIntermediateFactors(std::vector<uint32_t>& interFactors, std::vector<uint32_t> availableFactors, uint32_t tmpN, int curFactorsIndex) {
    while (curFactorsIndex >= 0) {
        while (tmpN % availableFactors[curFactorsIndex] == 0) {
            tmpN /= availableFactors[curFactorsIndex];
            interFactors.emplace_back(availableFactors[curFactorsIndex]);
        }
        curFactorsIndex -= 1;
    }
    while (interFactors.size() < MAX_FACTORS_LEN) {
        interFactors.emplace_back(1);
    }
}

static void CalculateFactors(uint32_t factors[], int64_t len, bool& isBluestein) {
    std::vector<uint32_t> availableFactors(LAST_FACTOR - FIRST_FACTOR + 1);
    std::iota(availableFactors.begin(), availableFactors.end(), FIRST_FACTOR);
    std::vector<uint32_t> factorsTmp, factorsTmpBluestein;
  
    int curFactorsIndex = availableFactors.size() - 1;
    uint32_t tmpN = len;
    
    if (tmpN == LAST_FACTOR * LAST_FACTOR * LAST_FACTOR * COMPLEX) {
        factorsTmp.emplace_back(LAST_FACTOR * COMPLEX);
        factorsTmp.emplace_back(LAST_FACTOR);
        factorsTmp.emplace_back(LAST_FACTOR);
        tmpN = 1;
    } else {
        CalculateIntermediateFactors(factorsTmp, availableFactors, tmpN, curFactorsIndex);
    }
    for (size_t i = 0; i < MAX_FACTORS_LEN; ++i) {
        factors[i] = factorsTmp[i];
    }

    isBluestein = (len % LAST_FACTOR != 0) || (factors[0] * factors[1] * factors[LAST_DIM] != len);
    
    if (isBluestein) {
        tmpN = uint32_t(POWER_OF_TWO * std::pow(POWER_OF_TWO, std::ceil(std::log2(double(len)))));
        curFactorsIndex = availableFactors.size() - 1;
        
        if (tmpN != LAST_FACTOR * LAST_FACTOR * LAST_FACTOR * COMPLEX) {
            CalculateIntermediateFactors(factorsTmpBluestein, availableFactors, tmpN, curFactorsIndex);
        } else {
            factorsTmpBluestein.emplace_back(LAST_FACTOR * COMPLEX);
            factorsTmpBluestein.emplace_back(LAST_FACTOR);
            factorsTmpBluestein.emplace_back(LAST_FACTOR);
            tmpN = 1;
        }
        for (size_t i = 0; i < MAX_FACTORS_LEN; ++i) {
            factors[i] = factorsTmpBluestein[i];
        }
    }
}

static void CalculationDft(std::vector<double>& dftRealCurVal, std::vector<double>& dftImagCurVal, 
                           std::vector<double>& dftRealBackCurVal, std::vector<double>& dftImagBackCurVal,
                           size_t curIndex, bool isBluestein, size_t colsNum, size_t curFactor, size_t& i, size_t& k) {
    for (size_t j = 0; j < colsNum / (COMPLEX - int(curIndex != 0)); ++j) {
        double param = curFactor != 0 ? -2. * M_PI * i * j / curFactor : 0;
        dftRealCurVal[k] = cos(param);
        dftRealBackCurVal[k] = cos(param);
        if (curIndex != 0) {
            dftImagCurVal[k] = sin(param);
            dftImagBackCurVal[k] = -sin(param);
        } else if (isBluestein) {
            dftRealCurVal[curFactor * colsNum + k] = -sin(param);
            dftRealBackCurVal[curFactor * colsNum + k] = sin(param);
        }                                   
        ++k;
        
        if (curIndex == 0) {
            dftRealCurVal[k] = sin(param);
            dftRealBackCurVal[k] = -sin(param);
            if (isBluestein) {
                dftRealCurVal[curFactor * colsNum + k] = cos(param);
                dftRealBackCurVal[curFactor * colsNum + k] = cos(param);
            }
            ++k;
        }
    }
}

static void CalculationMatricesValues(std::vector<double>& dftRealVal, std::vector<double>& dftImagVal,
                                      std::vector<double>& dftRealBackVal, std::vector<double>& dftImagBackVal,
                                      std::vector<double>& twiddleRealVal, std::vector<double>& twiddleImagVal,
                                      std::vector<double>& twiddleImagBackVal, std::vector<double>& dftRealCurVal,
                                      std::vector<double>& dftImagCurVal, std::vector<double>& dftRealBackCurVal,
                                      std::vector<double>& dftImagBackCurVal, std::vector<double>& twiddleRealCurVal,
                                      std::vector<double>& twiddleImagCurVal, std::vector<double>& twiddleImagBackCurVal,
                                      int64_t len, int64_t norm, size_t curIndex, bool isBluestein, size_t rowsNum, 
                                      size_t colsNum, size_t curFactor, size_t twiddleCurSize, size_t prevFactors) {
    size_t k = 0, l = 0;
    double normParam = 1.;
    if (norm != BACKWARD) {
        normParam = norm == FORWARD ? normParam / double(len) : normParam / sqrt(double(len));
    }
    for (size_t i = 0; i < rowsNum / (1 + int(curIndex == 0 && isBluestein)); ++i) {
        CalculationDft(dftRealCurVal, dftImagCurVal, dftRealBackCurVal, dftImagBackCurVal,
                       curIndex, isBluestein, colsNum, curFactor, i, k);

        if (curIndex != 0) {
            for (size_t j = 0; j < prevFactors; ++j) {
                double param = -2 * M_PI * i * j / (twiddleCurSize / 2.);
                twiddleRealCurVal[l] = cos(param);
                twiddleImagCurVal[l] = -sin(param);
                twiddleImagBackCurVal[l] = sin(param);
                ++l;

                twiddleRealCurVal[l] = cos(param);
                twiddleImagCurVal[l] = sin(param);
                twiddleImagBackCurVal[l] = -sin(param);
                ++l;
            }
        }                 
    }

    double pow2 = 2 * std::pow(2, std::ceil(std::log2(double(len))));
    for (size_t i = 0; i < dftRealCurVal.size(); ++i) {
        if (curIndex == 0) {
            dftRealVal.emplace_back(dftRealCurVal[i] * normParam);
            dftRealBackVal.emplace_back(dftRealBackCurVal[i] / pow2);
        } else {
            dftRealVal.emplace_back(dftRealCurVal[i]);
            dftRealBackVal.emplace_back(dftRealBackCurVal[i]);
        }
    }
    if (curIndex != 0) {
        for (size_t i = 0; i < dftImagCurVal.size(); ++i) {
            dftImagVal.emplace_back(dftImagCurVal[i]);
            dftImagBackVal.emplace_back(dftImagBackCurVal[i]);
        }
        for (size_t i = 0; i < twiddleRealCurVal.size(); ++i) {
            twiddleRealVal.emplace_back(twiddleRealCurVal[i]);
        }
        for (size_t i = 0; i < twiddleImagCurVal.size(); ++i) {
            twiddleImagVal.emplace_back(twiddleImagCurVal[i]);
            twiddleImagBackVal.emplace_back(twiddleImagBackCurVal[i]);
        }
    }
    dftRealCurVal.clear(), dftImagCurVal.clear();
    dftRealBackCurVal.clear(), dftImagBackCurVal.clear();
    twiddleRealCurVal.clear(), twiddleImagCurVal.clear(), twiddleImagBackCurVal.clear();
}

static void SetMatricesValues(std::vector<double> matrix, float* addrStart, size_t& i) {
    size_t j = 0;
    size_t temp = i;
    while (i < matrix.size() + temp) {
        *(addrStart + i) = float(matrix[j]);
        ++i;
        ++j;
    }
}

static const aclTensor* FinalCalculation(std::vector<double> dftRealVal, std::vector<double> dftImagVal,
                                         std::vector<double> twiddleRealVal, std::vector<double> twiddleImagVal,
                                         std::vector<double> dftRealBackVal, std::vector<double> dftImagBackVal,
                                         std::vector<double> twiddleImagBackVal, int64_t len, bool isBluestein,
                                         uint32_t& tensorLen, aclOpExecutor* executor) {
    std::vector<double> bluesteinRet; 
    if (isBluestein) {
        uint64_t pow2 = uint64_t(2 * std::pow(2, std::ceil(std::log2(double(len)))));
  	    bluesteinRet = GenerateAlphaBeta(len, pow2);
  	    tensorLen += dftRealBackVal.size() + dftImagBackVal.size() + twiddleRealVal.size() + twiddleImagBackVal.size() + bluesteinRet.size();
    }

    auto dftMatrix = executor->AllocHostTensor({tensorLen}, op::DataType::DT_FLOAT);
    float* addrStart = static_cast<float*>(dftMatrix->GetStorageAddr());

    for (size_t i = 0; i < tensorLen;) {
        SetMatricesValues(dftRealVal, addrStart, i);
        SetMatricesValues(dftImagVal, addrStart, i);
        SetMatricesValues(twiddleRealVal, addrStart, i);
        SetMatricesValues(twiddleImagVal, addrStart, i);
        
        if (isBluestein) {
            SetMatricesValues(dftRealBackVal, addrStart, i);
            SetMatricesValues(dftImagBackVal, addrStart, i);
            SetMatricesValues(twiddleRealVal, addrStart, i);
            SetMatricesValues(twiddleImagBackVal, addrStart, i);
            SetMatricesValues(bluesteinRet, addrStart, i);
        }
    }
    return dftMatrix;
}

static const aclTensor* GenerateTerminatorMatrix(int64_t len, int64_t norm, aclOpExecutor* executor) {
    auto deviceId = GetCurrentPlatformInfo().GetDeviceId();
    void* planDevice = Rfft1DSingleton::GetInstance().FindPlanCache(len, deviceId);
    if (planDevice != nullptr) {
        int64_t tensorLen = Rfft1DSingleton::GetInstance().FindTensorLen(len);
        auto dft = executor->AllocTensor({tensorLen}, op::DataType::DT_FLOAT);
        dft->SetFromWorkspace(false);
        dft->SetStorageAddr(planDevice);
        executor->AbandonCache();
        return dft;
    }

    std::vector<double> dftRealVal, dftImagVal, dftRealBackVal, dftImagBackVal,
                        twiddleRealVal, twiddleImagVal, twiddleImagBackVal;
  
    size_t prevFactors = 1;
    uint32_t factors[3] = {1, 1, 1};
    bool isBluestein = true;
    
    CalculateFactors(factors, len, isBluestein);

    for (size_t curIndex = 0; curIndex < MAX_FACTORS_LEN; ++curIndex) {
        size_t curFactor = factors[curIndex];
        size_t rowsNum = curFactor * (1 + int(curIndex == 0 && isBluestein));
        size_t colsNum = curFactor * (2 - int(curIndex != 0));
        size_t dftCurSize = rowsNum * colsNum;
        size_t twiddleCurSize = rowsNum * prevFactors * 2;
    
        std::vector<double> dftRealCurVal(dftCurSize, INIT_VALUE), dftImagCurVal(dftCurSize, INIT_VALUE);
        std::vector<double> dftRealBackCurVal(dftCurSize, INIT_VALUE), dftImagBackCurVal(dftCurSize, INIT_VALUE);    
        std::vector<double> twiddleRealCurVal(twiddleCurSize, INIT_VALUE), twiddleImagCurVal(twiddleCurSize, INIT_VALUE), twiddleImagBackCurVal(twiddleCurSize, INIT_VALUE);
    
        if (curFactor != 1) {
            CalculationMatricesValues(dftRealVal, dftImagVal, dftRealBackVal, dftImagBackVal, twiddleRealVal, 
                                      twiddleImagVal, twiddleImagBackVal, dftRealCurVal, dftImagCurVal, dftRealBackCurVal,
                                      dftImagBackCurVal, twiddleRealCurVal, twiddleImagCurVal, twiddleImagBackCurVal,
                                      len, norm, curIndex, isBluestein, rowsNum, colsNum, curFactor, twiddleCurSize, prevFactors);
        }    
      	prevFactors *= curFactor;
    }

    uint32_t tensorLen = dftRealVal.size() + dftImagVal.size() + twiddleRealVal.size() + twiddleImagVal.size();

    auto dftMatrix = FinalCalculation(dftRealVal, dftImagVal, twiddleRealVal, twiddleImagVal, dftRealBackVal, 
                                      dftImagBackVal, twiddleImagBackVal, len, isBluestein, tensorLen, executor);

    const aclTensor* deviceTensor = nullptr;
    auto deviceIdCacheNum = Rfft1DSingleton::GetInstance().FindCacheNum(deviceId);
    Rfft1DSingleton::GetInstance().AddTensorLen(len, tensorLen);
    if (deviceIdCacheNum < DEVICE_MAX_CACHE_NUM) {
        Rfft1DSingleton::GetInstance().AddCacheNum(deviceId);
        deviceTensor = op::CopyToNpuSync(dftMatrix, executor);
        CHECK_RET(deviceTensor != nullptr, nullptr);
        Rfft1DSingleton::GetInstance().AddPlanCache(len, deviceId, deviceTensor->GetData());
    } else {
        deviceTensor = op::CopyToNpu(dftMatrix, executor);
        CHECK_RET(deviceTensor != nullptr, nullptr);
    }

    return deviceTensor;
}

static const aclTensor* TransposeOutput(const aclTensor* out, int64_t dim, aclOpExecutor* executor) {
    auto outContiguous = l0op::Contiguous(out, executor);
    if (outContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TransposeOutput failed");
        return nullptr;
    }
    int64_t dims = outContiguous->GetViewShape().GetDimNum();
    
    if (dims > MIN_OUTPUT_DIMS) {
        std::vector<int64_t> valuePerm(dims);
        for (int64_t i = 0; i < dims; i++) 
        {
            valuePerm[i] = i;
        }
        
        std::swap(valuePerm[dim], valuePerm[dims - MIN_OUTPUT_DIMS]);
        auto perm = executor->AllocIntArray(valuePerm.data(), dims);
        outContiguous = l0op::Transpose(outContiguous, perm, executor);
    }
    return outContiguous;
}

static const aclTensor* GeneratePadInput(const aclTensor* self, int64_t n, int64_t lastDimLength, int64_t batches, aclOpExecutor* executor) {
    size_t dims = 2;
    int64_t left = 0;
    int64_t right = batches * (n - lastDimLength);
    std::vector<int64_t> padVec = {left, right};
    
    auto padArray = executor->AllocIntArray(padVec.data(), dims);
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc padVec failed");
        return nullptr;
    }
    
    auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);
    const aclTensor* valueTensor = executor->ConvertToTensor(executor->AllocScalar(PAD_VALUE), self->GetDataType());
    if (valueTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try convert PAD_VALUE pad tensor failed.");
        return nullptr;
    }
    
    return l0op::Pad(self, padTensor, executor);
}

static const aclTensor* InputProcessing(const aclTensor* selfContiguous, int64_t n, int64_t dims, aclOpExecutor* executor) {
    op::Shape newInputShape = selfContiguous->GetViewShape();
    int64_t lastDimLength = newInputShape.GetDim(dims - 1);
    if (n == -1) {
        n = lastDimLength;
    }

    op::Shape padShape;
    padShape.SetDimNum(dims);
    int64_t batches = 1;
    int64_t dimValue = 0;
    for (int i = 0; i < dims; i++) {
        batches = (i == (dims - 1)) ? batches : batches * newInputShape.GetDim(i);
        dimValue = (i == (dims - 1)) ? n : newInputShape.GetDim(i);
        padShape.SetDim(i, dimValue);
    }
    
    auto selfReshape = l0op::Reshape(selfContiguous, {batches, lastDimLength}, executor);
    int64_t valuePerm[2] = {1, 0};
    auto perm = executor->AllocIntArray(valuePerm, 2);
    
    auto selfTranspose = l0op::Transpose(selfReshape, perm, executor);
    auto selfTransposeReshape = l0op::Reshape(selfTranspose, {batches * lastDimLength}, executor);

    if (n < lastDimLength) {
        const int64_t offsetData[] = {0};
        aclIntArray* offsets = executor->AllocIntArray(offsetData, 1);
        const int64_t sizeData[] = {batches * n};
        aclIntArray* size = executor->AllocIntArray(sizeData, 1);
        auto selfPad = l0op::Slice(selfTransposeReshape, offsets, size, executor);
        
        auto selfPadReshape = l0op::Reshape(selfPad, {n, batches}, executor);
        auto selfSecondTranspose = l0op::Transpose(selfPadReshape, perm, executor);
        auto selfPreContiguous = l0op::Contiguous(selfSecondTranspose, executor);
        selfContiguous = l0op::Reshape(selfPreContiguous, padShape, executor);
    }
    if (n > lastDimLength) {
        auto selfPad = GeneratePadInput(selfTransposeReshape, n, lastDimLength, batches, executor);
        
        auto selfPadReshape = l0op::Reshape(selfPad, {n, batches}, executor);
        auto selfSecondTranspose = l0op::Transpose(selfPadReshape, perm, executor);
        auto selfPreContiguous = l0op::Contiguous(selfSecondTranspose, executor);
        selfContiguous = l0op::Reshape(selfPreContiguous, padShape, executor);
    }
    return selfContiguous;
}

aclnnStatus aclRfft1DGetWorkspaceSize(const aclTensor* self, int64_t n, int64_t dim, int64_t norm, 
                                      aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    L2_DFX_PHASE_1(aclRfft1D, DFX_IN(self, n, norm), DFX_OUT(out));
    OP_LOGD("Rfft1D: n %ld, norm %ld", n, norm);
    
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  
    bool result = CheckPlatform();
    CHECK_RET(result == true, ACLNN_ERR_PARAM_INVALID);

    if (HasEmptyTensor(self)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        OP_LOGD("self: nullptr, return");
        return ACLNN_SUCCESS;
    }
    
    op::Shape inputShape = self->GetViewShape();
    int64_t dims = inputShape.GetDimNum();

    auto ret = CheckParams(self, out, n, dim, norm, dims);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    dim = dim < 0 ? dim + dims : dim;
    
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (dims > 1) {
        std::vector<int64_t> valuePerm(dims);
    
        for (int64_t i = 0; i < dims; i++) {
            valuePerm[i] = i;
        }
        
        std::swap(valuePerm[dim], valuePerm[dims - 1]);
        auto perm = uniqueExecutor->AllocIntArray(valuePerm.data(), dims);
        selfContiguous = l0op::Transpose(selfContiguous, perm, uniqueExecutor.get());
    }
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    selfContiguous = InputProcessing(selfContiguous, n, dims, uniqueExecutor.get());
    selfContiguous = l0op::Contiguous(selfContiguous, uniqueExecutor.get());
    
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (!l0op::IsRfft1DAiCoreSupported(selfContiguous, n, norm)) {
        OP_LOGD("Rfft1D: aicpu");
    } else {
        OP_LOGD("Rfft1D: aicore");
    }

    const aclTensor* dftMatrix = n <= DFT_BORDER_VALUE ? GenerateDftMatrix(n, norm, uniqueExecutor.get()) :
                                                         GenerateTerminatorMatrix(n, norm, uniqueExecutor.get());
    
    auto outResult = l0op::Rfft1D(selfContiguous, dftMatrix, n, norm, uniqueExecutor.get());
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto rfft1dResult = TransposeOutput(outResult, dim, uniqueExecutor.get());
    
    CHECK_RET(rfft1dResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckReduceOutShape(rfft1dResult, out), ACLNN_ERR_PARAM_INVALID);
    auto viewCopyResult = l0op::ViewCopy(rfft1dResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclRfft1D(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclRfft1D);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}