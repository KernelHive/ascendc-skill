/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>

#include "acl/acl.h"
#include "aclnn_complex_mat_mul.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

bool ReadFile(const std::string &filePath, size_t fileSize, void *buffer, size_t bufferSize) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }
    file.read(static_cast<char*>(buffer), bufferSize);
    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size) {
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }
    file.write(static_cast<const char*>(buffer), size);
    file.close();
    return true;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) shapeSize *= i;
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclInit failed. ERROR: %d", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtSetDevice failed. ERROR: %d", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtCreateStream failed. ERROR: %d", ret); return FAILED);
    return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::string &filePath, const std::vector<int64_t> &shape, 
                    void **deviceAddr, aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtMalloc failed. ERROR: %d", ret); return FAILED);
    
    std::vector<T> hostData(GetShapeSize(shape));
    if (!ReadFile(filePath, 0, hostData.data(), size)) {
        return FAILED;
    }
    
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtMemcpy failed. ERROR: %d", ret); return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, 
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return SUCCESS;
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == SUCCESS, return FAILED);

    // Input shapes: [batch, m, k] for x, [batch, k, n] for y, [batch, m, n] for bias
    std::vector<int64_t> inputXShape = {4, 300, 400};
    std::vector<int64_t> inputYShape = {4, 400, 500};
    std::vector<int64_t> biasShape = {4, 300, 500};
    std::vector<int64_t> outputShape = {4, 300, 500};

    // Create tensors
    void *inputXDevice = nullptr, *inputYDevice = nullptr, *biasDevice = nullptr, *outputDevice = nullptr;
    aclTensor *inputX = nullptr, *inputY = nullptr, *bias = nullptr, *output = nullptr;
    
    ret = CreateAclTensor<std::complex<float>>("../input/input_x.bin", inputXShape, &inputXDevice, 
                                              ACL_COMPLEX64, &inputX);
    CHECK_RET(ret == SUCCESS, return FAILED);
    ret = CreateAclTensor<std::complex<float>>("../input/input_y.bin", inputYShape, &inputYDevice, 
                                              ACL_COMPLEX64, &inputY);
    CHECK_RET(ret == SUCCESS, return FAILED);
    ret = CreateAclTensor<std::complex<float>>("../input/input_bias.bin", biasShape, &biasDevice, 
                                              ACL_COMPLEX64, &bias);
    CHECK_RET(ret == SUCCESS, return FAILED);
    
    // Create output tensor
    auto outputSize = GetShapeSize(outputShape) * sizeof(std::complex<float>);
    ret = aclrtMalloc(&outputDevice, outputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtMalloc for output failed. ERROR: %d", ret); return FAILED);
    output = aclCreateTensor(outputShape.data(), outputShape.size(), ACL_COMPLEX64, nullptr, 0, 
                            ACL_FORMAT_ND, outputShape.data(), outputShape.size(), outputDevice);

    // Get workspace size
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    ret = aclnnComplexMatMulGetWorkspaceSize(inputX, inputY, bias, output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclnnComplexMatMulGetWorkspaceSize failed. ERROR: %d", ret); return FAILED);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("Workspace allocation failed. ERROR: %d", ret); return FAILED);
    }

    // Execute operator
    ret = aclnnComplexMatMul(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclnnComplexMatMul failed. ERROR: %d", ret); return FAILED);

    // Synchronize and copy result
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtSynchronizeStream failed. ERROR: %d", ret); return FAILED);

    std::vector<std::complex<float>> result(GetShapeSize(outputShape));
    ret = aclrtMemcpy(result.data(), outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("Result copy failed. ERROR: %d", ret); return FAILED);

    WriteFile("../output/output_z.bin", result.data(), outputSize);
    INFO_LOG("Write output success");

    // Cleanup
    aclDestroyTensor(inputX);
    aclDestroyTensor(inputY);
    aclDestroyTensor(bias);
    aclDestroyTensor(output);
    aclrtFree(inputXDevice);
    aclrtFree(inputYDevice);
    aclrtFree(biasDevice);
    aclrtFree(outputDevice);
    if (workspace) { aclrtFree(workspace); }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return SUCCESS;
}
