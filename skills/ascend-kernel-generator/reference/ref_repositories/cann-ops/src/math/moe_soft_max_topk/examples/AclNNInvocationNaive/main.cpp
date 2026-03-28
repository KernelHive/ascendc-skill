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
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)


#include "acl/acl.h"
#include "aclnn_moe_soft_max_topk.h"


#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

    
bool ReadFile(const std::string &filePath, size_t fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
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
    // 固定写法，acl初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);

    return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed. ERROR: %d\n", ret); return FAILED);
    return SUCCESS;
}

int main(int argc, char **argv)
{
    // 0. 新增参数校验
    if (argc < 2) {
        LOG_PRINT("Usage: %s [case1|case2]\n", argv[0]);
        return FAILED;
    }
    // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputXShape;

    int64_t k = 0;
    if (strcmp(argv[1], "case1") == 0) {
        inputXShape = {1024, 16};  // case1的原始shapeX
        k = 4;
    } else if (strcmp(argv[1], "case2") == 0) {
        inputXShape = {2048, 32};  // case2的新shapeX
        k = 6;
    } else {
        LOG_PRINT("Invalid case argument. Please use case1 or case2.\n");
        return FAILED;
    }

    std::vector<int64_t> outputYShape = {inputXShape[0], k};
    std::vector<int64_t> outputIndicesShape = {inputXShape[0], k};

    void *inputXDeviceAddr = nullptr;
    void *outputYDeviceAddr = nullptr;
    void *outputIndicesDeviceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *outputY = nullptr;
    aclTensor *outputIndices = nullptr;
    size_t inputXShapeSize_1 = inputXShape[0] * inputXShape[1];
    size_t outputYShapeSize_1 = outputYShape[0] * outputYShape[1];
    size_t outputIndicesShapeSize_1 = outputIndicesShape[0] * outputIndicesShape[1];
    size_t dataTypeSize = 4; // float and int32_t are 4 bytes
    std::vector<float> inputXHostData(inputXShape[0] * inputXShape[1]);
    std::vector<float> outputYHostData(outputYShape[0] * outputYShape[1]);
    std::vector<int32_t> outputIndicesHostData(outputIndicesShape[0] * outputIndicesShape[1]);

    size_t fileSize = 0;
    void** input1=(void**)(&inputXHostData);

    //读取数据
    ReadFile("../input/input_x.bin", fileSize, *input1, inputXShapeSize_1 * dataTypeSize);
    INFO_LOG("Set input success");

    // 创建inputX aclTensor
    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建outputZ aclTensor
    ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, aclDataType::ACL_FLOAT, &outputY);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建outputZ aclTensor
    ret = CreateAclTensor(outputIndicesHostData, outputIndicesShape, &outputIndicesDeviceAddr, aclDataType::ACL_INT32, &outputIndices);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    // 3. 调用CANN自定义算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 计算workspace大小并申请内存
    ret = aclnnMoeSoftMaxTopkGetWorkspaceSize(inputX, k, outputY, outputIndices, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeSoftMaxTopkGetWorkspaceSize failed. ERROR: %d\n", ret); return FAILED);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
    }
    // 执行算子
    ret = aclnnMoeSoftMaxTopk(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeSoftMaxTopk failed. ERROR: %d\n", ret); return FAILED);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto sizeY = GetShapeSize(outputYShape);
    auto sizeIndices = GetShapeSize(outputIndicesShape);
    std::vector<float> resultYData(sizeY, 0);
    std::vector<float> resultIndicesData(sizeIndices, 0);
    ret = aclrtMemcpy(resultYData.data(), resultYData.size() * sizeof(resultYData[0]), outputYDeviceAddr,
                      sizeY * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result Y from device to host failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtMemcpy(resultIndicesData.data(), resultIndicesData.size() * sizeof(resultIndicesData[0]), outputIndicesDeviceAddr,
                      sizeIndices * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result Indices from device to host failed. ERROR: %d\n", ret); return FAILED);
    void** output1=(void**)(&resultYData);
    void** output2=(void**)(&resultIndicesData);

    //写出数据
    WriteFile("../output/output_y.bin", *output1, outputYShapeSize_1 * dataTypeSize);
    INFO_LOG("Write output Y success");
    WriteFile("../output/output_indices.bin", *output2, outputIndicesShapeSize_1 * dataTypeSize);
    INFO_LOG("Write output Indices success");

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(inputX);
    aclDestroyTensor(outputY);
    aclDestroyTensor(outputIndices);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(inputXDeviceAddr);
    aclrtFree(outputYDeviceAddr);
    aclrtFree(outputIndicesDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return SUCCESS;
}