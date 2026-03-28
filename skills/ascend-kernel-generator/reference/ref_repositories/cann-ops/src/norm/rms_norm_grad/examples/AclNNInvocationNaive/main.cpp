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
#include "aclnn_rms_norm_grad.h"

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
    (void)close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);

    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_NCDHW,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> gradInputShape = {2, 1, 16};
    std::vector<int64_t> xInputShape = {2, 1, 16};
    std::vector<int64_t> rstdInputShape = {2};
    std::vector<int64_t> gammaInputShape = {16};
    std::vector<int64_t> dxOutputShape = {2, 1, 16};
    std::vector<int64_t> dgammaOutputShape = {16};

    void *gradInputDeviceAddr = nullptr;
    void *xInputDeviceAddr = nullptr;
    void *rstdInputDeviceAddr = nullptr;
    void *gammaInputDeviceAddr = nullptr;
    void *dxOutDeviceAddr = nullptr;
    void *dgammaOutDeviceAddr = nullptr;

    aclTensor *gradInput = nullptr;
    aclTensor *xInput = nullptr;
    aclTensor *rstdInput = nullptr;
    aclTensor *gammaInput = nullptr;
    aclTensor *dxOut = nullptr;
    aclTensor *dgammaOut = nullptr;

    std::vector<float> gradInputHostData = {1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32};
    std::vector<float> xInputHostData = {1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32};
    std::vector<float> rstdInputHostData = {1, 2};
    std::vector<float> gammaInputHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    std::vector<float> dxOutHostData = {1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32};
    std::vector<float> dgammaOutHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    std::vector<int64_t> output1SizeData = {2, 1, 16};
    std::vector<int64_t> output2SizeData = {16};
    std::vector<int64_t> input1SizeData = {2, 1, 16};
    std::vector<int64_t> input2SizeData = {2};
    std::vector<int64_t> input3SizeData = {16};

    ret = CreateAclTensor(gradInputHostData, input1SizeData, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(xInputHostData, input1SizeData, &xInputDeviceAddr, aclDataType::ACL_FLOAT, &xInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(rstdInputHostData, input2SizeData, &rstdInputDeviceAddr, aclDataType::ACL_FLOAT, &rstdInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret =
        CreateAclTensor(gammaInputHostData, input3SizeData, &gammaInputDeviceAddr, aclDataType::ACL_FLOAT, &gammaInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(dxOutHostData, output1SizeData, &dxOutDeviceAddr, aclDataType::ACL_FLOAT, &dxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(dgammaOutHostData, output2SizeData, &dgammaOutDeviceAddr, aclDataType::ACL_FLOAT, &dgammaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnRmsNormGrad第一段接口
    ret = aclnnRmsNormGradGetWorkspaceSize(
        gradInput, xInput, rstdInput, gammaInput, dxOut, dgammaOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnRmsNormGrad第二段接口
    ret = aclnnRmsNormGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGrad failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size_dx = GetShapeSize(gradInputShape);
    std::vector<float> resultData1(size_dx, 0);
    ret = aclrtMemcpy(resultData1.data(),
        resultData1.size() * sizeof(resultData1[0]),
        dxOutDeviceAddr,
        size_dx * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size_dx; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData1[i]);
    }
    // 写出数据
    void **output_out2=(void **)(&resultData1);
    WriteFile("../output/output_out2.bin", *output_out2, size_dx * sizeof(resultData1[0]));
    INFO_LOG("Write dx output success");
    
    auto size_dgamma = GetShapeSize(gammaInputShape);
    std::vector<float> resultData2(size_dgamma, 1);
    ret = aclrtMemcpy(resultData2.data(),
        resultData2.size() * sizeof(resultData2[0]),
        dgammaOutDeviceAddr,
        size_dgamma * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size_dgamma; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData2[i]);
    }
    // 写出数据
    void **output_out=(void **)(&resultData2);
    WriteFile("../output/output_out1.bin", *output_out, size_dgamma * sizeof(resultData2[0]));
    INFO_LOG("Write dgamma output success");

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradInput);
    aclDestroyTensor(xInput);
    aclDestroyTensor(rstdInput);
    aclDestroyTensor(gammaInput);
    aclDestroyTensor(dxOut);
    aclDestroyTensor(dgammaOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradInputDeviceAddr);
    aclrtFree(xInputDeviceAddr);
    aclrtFree(rstdInputDeviceAddr);
    aclrtFree(gammaInputDeviceAddr);
    aclrtFree(dxOutDeviceAddr);
    aclrtFree(dgammaOutDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
