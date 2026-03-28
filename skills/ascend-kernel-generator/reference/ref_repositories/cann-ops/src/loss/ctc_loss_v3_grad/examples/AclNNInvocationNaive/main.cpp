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
#include "aclnn_ctc_loss_backward.h"

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
    return SUCCESS;
}

int main(int argc, char **argv)
{
    // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t C = 10000;
    int64_t N = 20;
    int64_t S = 10;
    int64_t T = 12;
    int64_t alpha = 2 * S + 1;
    std::vector<int64_t> gradOutShape = {N};
    std::vector<int64_t> logProbsShape = {T, N, C};
    std::vector<int64_t> targetsShape = {N, S};
    std::vector<int64_t> logAlphaShape = {N, T, alpha};
    std::vector<int64_t> outShape = logProbsShape;
    void* gradOutDeviceAddr = nullptr;
    void* logProbsDeviceAddr = nullptr;
    void* targetsDeviceAddr = nullptr;
    void* negLoglikelihoodDeviceAddr = nullptr;
    void* logAlphaDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;

    aclTensor* gradOut = nullptr;
    aclTensor* logProbs = nullptr;
    aclTensor* targets = nullptr;
    aclIntArray* inputLengths = nullptr;
    aclIntArray* targetLengths = nullptr;
    aclTensor* negLoglikelihood = nullptr;
    aclTensor* logAlpha = nullptr;
    aclTensor* out = nullptr;

    size_t gradOutShapeSize = gradOutShape[0];
    size_t logProbsShapeSize = logProbsShape[0] * logProbsShape[1] * logProbsShape[2];
    size_t targetsShapeSize = targetsShape[0] * targetsShape[1];
    size_t logAlphaShapeSize = logAlphaShape[0] * logAlphaShape[1] * logAlphaShape[2];
    size_t outShapeSize = logProbsShapeSize;


    std::vector<float> gradOutHostData(gradOutShapeSize);
    std::vector<float> logProbsHostData(logProbsShapeSize);
    std::vector<int64_t> targetsHostData(targetsShapeSize);
    std::vector<int64_t> inputLengthsSizeData(gradOutShapeSize);
    std::vector<int64_t> targetLengthsSizeData(gradOutShapeSize);
    std::vector<float> negLoglikelihoodHostData(gradOutShapeSize);
    std::vector<float> logAlphaHostData(logAlphaShapeSize);

    std::vector<float> outHostData(outShapeSize);

    size_t dataType = sizeof(int32_t);
    size_t fileSize = 0;
    void ** inputGradOut=(void **)(&gradOutHostData);
    void ** inputLogProb=(void **)(&logProbsHostData);
    void ** inputTargets=(void **)(&targetsHostData);

    void ** inputInputLengths=(void **)(&inputLengthsSizeData);
    void ** inputTargetLengths=(void **)(&targetLengthsSizeData);

    void ** inputLogAlpha=(void **)(&logAlphaHostData);
    void ** inputNegLoss=(void **)(&negLoglikelihoodHostData);

    //读取数据
    ReadFile("../input/input_grad_out.bin", fileSize, *inputGradOut, gradOutShapeSize * dataType);
    ReadFile("../input/input_log_probs.bin", fileSize, *inputLogProb, logProbsShapeSize * dataType);
    ReadFile("../input/input_targets.bin", fileSize, *inputTargets, targetsShapeSize * sizeof(int64_t));
    ReadFile("../input/input_input_lengths.bin", fileSize, *inputInputLengths, gradOutShapeSize * sizeof(int64_t));
    ReadFile("../input/input_target_lengths.bin", fileSize, *inputTargetLengths, gradOutShapeSize * sizeof(int64_t));
    ReadFile("../input/input_log_alpha.bin", fileSize, *inputLogAlpha, logAlphaShapeSize * dataType);
    ReadFile("../input/input_neg_log_likelihood.bin", fileSize, *inputNegLoss, gradOutShapeSize * dataType);

    

    INFO_LOG("Set input success");
    // 创建inputX aclTensor
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    ret = CreateAclTensor(logProbsHostData, logProbsShape, &logProbsDeviceAddr, aclDataType::ACL_FLOAT, &logProbs);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建inputX aclTensor
    ret = CreateAclTensor(targetsHostData, targetsShape, &targetsDeviceAddr, aclDataType::ACL_INT64, &targets);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    ret = CreateAclTensor(negLoglikelihoodHostData, gradOutShape, &negLoglikelihoodDeviceAddr, aclDataType::ACL_FLOAT, &negLoglikelihood);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建inputX aclTensor
    ret = CreateAclTensor(logAlphaHostData, logAlphaShape, &logAlphaDeviceAddr, aclDataType::ACL_FLOAT, &logAlpha);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    inputLengths = aclCreateIntArray(inputLengthsSizeData.data(), gradOutShapeSize);
    CHECK_RET(inputLengths != nullptr, return ACL_ERROR_BAD_ALLOC);
    targetLengths = aclCreateIntArray(targetLengthsSizeData.data(), gradOutShapeSize);
    CHECK_RET(targetLengths != nullptr, return ACL_ERROR_BAD_ALLOC);

    // 创建outputZ aclTensor
    ret = CreateAclTensor(outHostData, logProbsShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    // 3. 调用CANN自定义算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 计算workspace大小并申请内存
    ret = aclnnCtcLossBackwardGetWorkspaceSize(gradOut, logProbs, targets, inputLengths, targetLengths,
                                                negLoglikelihood, logAlpha, 0, false, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLossBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return FAILED);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
    }
    // 执行算子
    ret = aclnnCtcLossBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLossBackward failed. ERROR: %d\n", ret); return FAILED);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    std::vector<float> resultData(logProbsShapeSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
    void ** output1=(void **)(&resultData);
    //写出数据
    WriteFile("../output/output.bin", *output1, logProbsShapeSize * dataType);
    INFO_LOG("Write output success");

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOut);
    aclDestroyTensor(logProbs);
    aclDestroyTensor(targets);
    aclDestroyIntArray(inputLengths);
    aclDestroyIntArray(targetLengths);
    aclDestroyTensor(negLoglikelihood);
    aclDestroyTensor(logAlpha);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(logProbsDeviceAddr);
    aclrtFree(targetsDeviceAddr);
    aclrtFree(negLoglikelihoodDeviceAddr);
    aclrtFree(logAlphaDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return SUCCESS;
}
