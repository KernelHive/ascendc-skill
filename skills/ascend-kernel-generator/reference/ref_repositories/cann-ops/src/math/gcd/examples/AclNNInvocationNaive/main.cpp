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
 * @file main.cpp
 */

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>

#include "acl/acl.h"
#include "aclnn_gcd.h"

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

bool ReadFile(const std::string &filePath, void *buffer, size_t bufferSize)
{
    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    file.read(static_cast<char *>(buffer), bufferSize);
    if (!file) {
        ERROR_LOG("Read file failed. path = %s", filePath.c_str());
        file.close();
        return false;
    }
    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void) close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclInit failed. ERROR: %d", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtSetDevice failed. ERROR: %d", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtCreateStream failed. ERROR: %d", ret); return FAILED);
    return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, 
                    aclDataType dataType, aclTensor **tensor, void **deviceAddr)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtMalloc failed. ERROR: %d", ret); return FAILED);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtMemcpy failed. ERROR: %d", ret); return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, 
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, ERROR_LOG("aclCreateTensor failed"); return FAILED);
    return SUCCESS;
}

int main(int argc, char **argv)
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == SUCCESS, return FAILED);

    std::vector<int64_t> inputX1Shape = {1024, 1024};
    std::vector<int64_t> inputX2Shape = {1024, 1024};
    std::vector<int64_t> outputYShape = {1024, 1024};
    
    const size_t elementCount = 1024UL * 1024UL;
    const size_t dataSize = elementCount * sizeof(int32_t);
    
    std::vector<int32_t> inputX1HostData(elementCount);
    std::vector<int32_t> inputX2HostData(elementCount);
    std::vector<int32_t> outputYHostData(elementCount);

    // Read input data
    CHECK_RET(ReadFile("../input/input_x1.bin", inputX1HostData.data(), dataSize), 
              return FAILED);
    CHECK_RET(ReadFile("../input/input_x2.bin", inputX2HostData.data(), dataSize), 
              return FAILED);
    INFO_LOG("Read input success");

    // Create ACL tensors
    void *inputX1DeviceAddr = nullptr;
    void *inputX2DeviceAddr = nullptr;
    void *outputYDeviceAddr = nullptr;
    aclTensor *inputX1 = nullptr;
    aclTensor *inputX2 = nullptr;
    aclTensor *outputY = nullptr;
    
    ret = CreateAclTensor(inputX1HostData, inputX1Shape, ACL_INT32, &inputX1, &inputX1DeviceAddr);
    CHECK_RET(ret == SUCCESS, return FAILED);
    ret = CreateAclTensor(inputX2HostData, inputX2Shape, ACL_INT32, &inputX2, &inputX2DeviceAddr);
    CHECK_RET(ret == SUCCESS, return FAILED);
    ret = CreateAclTensor(outputYHostData, outputYShape, ACL_INT32, &outputY, &outputYDeviceAddr);
    CHECK_RET(ret == SUCCESS, return FAILED);

    // Call GCD operator
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnGcdGetWorkspaceSize(inputX1, inputX2, outputY, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclnnGcdGetWorkspaceSize failed. ERROR: %d", ret); return FAILED);
    
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("allocate workspace failed. ERROR: %d", ret); return FAILED);
    }
    
    ret = aclnnGcd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclnnGcd failed. ERROR: %d", ret); return FAILED);
    
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("aclrtSynchronizeStream failed. ERROR: %d", ret); return FAILED);

    // Copy output from device
    ret = aclrtMemcpy(outputYHostData.data(), dataSize, outputYDeviceAddr, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, ERROR_LOG("copy result from device to host failed. ERROR: %d", ret); return FAILED);
    
    // Write output
    CHECK_RET(WriteFile("../output/output_y.bin", outputYHostData.data(), dataSize), 
              return FAILED);
    INFO_LOG("Write output success");

    // Cleanup
    aclDestroyTensor(inputX1);
    aclDestroyTensor(inputX2);
    aclDestroyTensor(outputY);
    aclrtFree(inputX1DeviceAddr);
    aclrtFree(inputX2DeviceAddr);
    aclrtFree(outputYDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return SUCCESS;
}
