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
#include "aclnn_add_layer_norm.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
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
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造，本示例中将各调用一次不带bias可选输入的和带bias输入的用例
    float eps = 1e-6;
    bool additionalOut = true;

    std::vector<int64_t> x1Shape = {1, 2, 8};
    std::vector<int64_t> x2Shape = {1, 2, 8};
    std::vector<int64_t> gammaShape = {8};
    std::vector<int64_t> betaShape = {8};
    std::vector<int64_t> biasShape = {8};

    std::vector<int64_t> outputYShape = {1, 2, 8};
    std::vector<int64_t> outputMeanShape = {1, 2, 1};
    std::vector<int64_t> outputRstdShape = {1, 2, 1};
    std::vector<int64_t> outputXShape = {1, 2, 8};

    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *betaDeviceAddr = nullptr;
    void *gammaDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;

    // 用于不带bias的输出 Device 地址
    void *outputYDeviceAddr = nullptr;
    void *outputMeanDeviceAddr = nullptr;
    void *outputRstdDeviceAddr = nullptr;
    void *outputXDeviceAddr = nullptr;

    // 用于带bias的输出 Device 地址
    void *outputYDeviceAddrBias = nullptr;
    void *outputMeanDeviceAddrBias = nullptr;
    void *outputRstdDeviceAddrBias = nullptr;
    void *outputXDeviceAddrBias = nullptr;

    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *beta = nullptr;
    aclTensor *gamma = nullptr;
    aclTensor *bias = nullptr;

    // 用于不带bias的aclTensor
    aclTensor *outputY = nullptr;
    aclTensor *outputMean = nullptr;
    aclTensor *outputRstd = nullptr;
    aclTensor *outputX = nullptr;

    // 用于带bias的aclTensor
    aclTensor *outputYBias = nullptr;
    aclTensor *outputMeanBias = nullptr;
    aclTensor *outputRstdBias = nullptr;
    aclTensor *outputXBias = nullptr;

    std::vector<float> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> x2HostData = {4, 4, 4, 4, 4, 4, 4, 4, -3, -3, -3, -3, -3, -3, -3, -3};
    std::vector<float> gammaHostData = {2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> betaHostData = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    std::vector<float> biasHostData = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    // 用于不带bias的HostData
    std::vector<float> outputYHostData(1 * 2 * 8);
    std::vector<float> outputMeanHostData(2);
    std::vector<float> outputRstdHostData(2);
    std::vector<float> outputXHostData(1 * 2 * 8);

    // 用于带bias的HostData
    std::vector<float> outputYHostDataBias(1 * 2 * 8);
    std::vector<float> outputMeanHostDataBias(2);
    std::vector<float> outputRstdHostDataBias(2);
    std::vector<float> outputXHostDataBias(1 * 2 * 8);

    // 创建self aclTensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建不带 bias 的 aclTensor
    ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, aclDataType::ACL_FLOAT, &outputY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputMeanHostData, outputMeanShape, &outputMeanDeviceAddr, aclDataType::ACL_FLOAT, &outputMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputRstdHostData, outputRstdShape, &outputRstdDeviceAddr, aclDataType::ACL_FLOAT, &outputRstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outputXHostData, outputXShape, &outputXDeviceAddr, aclDataType::ACL_FLOAT, &outputX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建带 bias 的 aclTensor
    ret = CreateAclTensor(
        outputYHostDataBias, outputYShape, &outputYDeviceAddrBias, aclDataType::ACL_FLOAT, &outputYBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputMeanHostDataBias, outputMeanShape, &outputMeanDeviceAddrBias, aclDataType::ACL_FLOAT, &outputMeanBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputRstdHostDataBias, outputRstdShape, &outputRstdDeviceAddrBias, aclDataType::ACL_FLOAT, &outputRstdBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputXHostDataBias, outputXShape, &outputXDeviceAddrBias, aclDataType::ACL_FLOAT, &outputXBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnnAddLayerNorm接口调用示例，包含带bias和不带bias的各一次
    // 3. 调用CANN算子库API，需要修改为具体的Api名称

    // 3.1 不带bias可选输入的示例
    // 调用aclnnAddLayerNorm第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    LOG_PRINT("\nUse aclnnAddLayerNorm Non-Bias Port.");
    // bias参数直接传入nullptr即可
    ret = aclnnAddLayerNormGetWorkspaceSize(x1,
        x2,
        gamma,
        beta,
        nullptr,
        eps,
        additionalOut,
        outputY,
        outputMean,
        outputRstd,
        outputX,
        &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAddLayerNorm第二段接口
    ret = aclnnAddLayerNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNorm failed. ERROR: %d\n", ret); return ret);

    // 3.2 带bias可选输入的示例
    // 调用aclnnAddLayerNorm第一段接口
    uint64_t workspaceSizeBias = 0;
    aclOpExecutor *executorBias;
    LOG_PRINT("\nUse aclnnAddLayerNorm Bias Port.");
    // 正常传入bias即可
    ret = aclnnAddLayerNormGetWorkspaceSize(x1,
        x2,
        gamma,
        beta,
        bias,
        eps,
        additionalOut,
        outputYBias,
        outputMeanBias,
        outputRstdBias,
        outputXBias,
        &workspaceSizeBias,
        &executorBias);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddrBias = nullptr;
    if (workspaceSizeBias > 0) {
        ret = aclrtMalloc(&workspaceAddrBias, workspaceSizeBias, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAddLayerNorm第二段接口
    ret = aclnnAddLayerNorm(workspaceAddrBias, workspaceSizeBias, executorBias, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNorm failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改

    // 5.1 考出不带bias的输出
    auto outputYSize = GetShapeSize(outputYShape);
    std::vector<float> resultDataY(outputYSize, 0);
    ret = aclrtMemcpy(resultDataY.data(),
        resultDataY.size() * sizeof(resultDataY[0]),
        outputYDeviceAddr,
        outputYSize * sizeof(resultDataY[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-bias: y output");
    for (int64_t i = 0; i < outputYSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataY[i]);
    }
    // 写出数据
    void **output_out = (void **)(&resultDataY);
    WriteFile("../output/output_y.bin", *output_out, outputYSize * sizeof(resultDataY[0]));

    auto outputMeanSize = GetShapeSize(outputMeanShape);
    std::vector<float> resultDataMean(outputMeanSize, 0);
    ret = aclrtMemcpy(resultDataMean.data(),
        resultDataMean.size() * sizeof(resultDataMean[0]),
        outputMeanDeviceAddr,
        outputMeanSize * sizeof(resultDataMean[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-bias: mean output");
    for (int64_t i = 0; i < outputMeanSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataMean[i]);
    }
    // 写出数据
    void **output_out_mean = (void **)(&resultDataMean);
    WriteFile("../output/output_mean.bin", *output_out_mean, outputMeanSize * sizeof(resultDataMean[0]));

    auto outputRstdSize = GetShapeSize(outputRstdShape);
    std::vector<float> resultDataRstd(outputRstdSize, 0);
    ret = aclrtMemcpy(resultDataRstd.data(),
        resultDataRstd.size() * sizeof(resultDataRstd[0]),
        outputRstdDeviceAddr,
        outputRstdSize * sizeof(resultDataRstd[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-bias: rstd output");
    for (int64_t i = 0; i < outputRstdSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataRstd[i]);
    }
    // 写出数据
    void **output_out_rstd = (void **)(&resultDataRstd);
    WriteFile("../output/output_rstd.bin", *output_out_rstd, outputRstdSize * sizeof(resultDataRstd[0]));

    auto outputXSize = GetShapeSize(outputXShape);
    std::vector<float> resultDataX(outputXSize, 0);
    ret = aclrtMemcpy(resultDataX.data(),
        resultDataX.size() * sizeof(resultDataX[0]),
        outputXDeviceAddr,
        outputXSize * sizeof(resultDataX[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-bias: x output");
    for (int64_t i = 0; i < outputXSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataX[i]);
    }
    // 写出数据
    void **output_out_x = (void **)(&resultDataX);
    WriteFile("../output/output_x.bin", *output_out_x, outputXSize * sizeof(resultDataX[0]));

    // 5.2 考出带bias的输出
    auto outputYSizeBias = GetShapeSize(outputYShape);
    std::vector<float> resultDataYBias(outputYSizeBias, 0);
    ret = aclrtMemcpy(resultDataYBias.data(),
        resultDataYBias.size() * sizeof(resultDataYBias[0]),
        outputYDeviceAddrBias,
        outputYSizeBias * sizeof(resultDataYBias[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm bias: y output");
    for (int64_t i = 0; i < outputYSizeBias; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataYBias[i]);
    }
    // 写出数据
    void **output_out_ybias = (void **)(&resultDataYBias);
    WriteFile("../output/output_ybias.bin", *output_out_ybias, outputYSizeBias * sizeof(resultDataYBias[0]));

    auto outputMeanSizeBias = GetShapeSize(outputMeanShape);
    std::vector<float> resultDataMeanBias(outputMeanSizeBias, 0);
    ret = aclrtMemcpy(resultDataMeanBias.data(),
        resultDataMeanBias.size() * sizeof(resultDataMeanBias[0]),
        outputMeanDeviceAddrBias,
        outputMeanSizeBias * sizeof(resultDataMeanBias[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm bias: mean output");
    for (int64_t i = 0; i < outputMeanSizeBias; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataMeanBias[i]);
    }
    // 写出数据
    void **output_meanbias = (void **)(&resultDataMeanBias);
    WriteFile("../output/output_meanbias.bin", *output_meanbias, outputMeanSizeBias * sizeof(resultDataMeanBias[0]));

    auto outputRstdSizeBias = GetShapeSize(outputRstdShape);
    std::vector<float> resultDataRstdBias(outputRstdSizeBias, 0);
    ret = aclrtMemcpy(resultDataRstdBias.data(),
        resultDataRstdBias.size() * sizeof(resultDataRstdBias[0]),
        outputRstdDeviceAddrBias,
        outputRstdSizeBias * sizeof(resultDataRstdBias[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm bias: rstd output");
    for (int64_t i = 0; i < outputRstdSizeBias; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataRstdBias[i]);
    }
    // 写出数据
    void **output_rstdbias = (void **)(&resultDataRstdBias);
    WriteFile("../output/output_rstdbias.bin", *output_rstdbias, outputRstdSizeBias * sizeof(resultDataRstdBias[0]));

    auto outputXSizeBias = GetShapeSize(outputXShape);
    std::vector<float> resultDataXBias(outputXSizeBias, 0);
    ret = aclrtMemcpy(resultDataXBias.data(),
        resultDataXBias.size() * sizeof(resultDataXBias[0]),
        outputXDeviceAddrBias,
        outputXSizeBias * sizeof(resultDataXBias[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm bias: x output");
    for (int64_t i = 0; i < outputXSizeBias; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataXBias[i]);
    }
    // 写出数据
    void **output_xbias = (void **)(&resultDataXBias);
    WriteFile("../output/output_xbias.bin", *output_xbias, outputXSizeBias * sizeof(resultDataXBias[0]));

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(beta);
    aclDestroyTensor(gamma);
    aclDestroyTensor(bias);

    aclDestroyTensor(outputY);
    aclDestroyTensor(outputMean);
    aclDestroyTensor(outputRstd);
    aclDestroyTensor(outputX);

    aclDestroyTensor(outputYBias);
    aclDestroyTensor(outputMeanBias);
    aclDestroyTensor(outputRstdBias);
    aclDestroyTensor(outputXBias);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(biasDeviceAddr);

    aclrtFree(outputYDeviceAddr);
    aclrtFree(outputMeanDeviceAddr);
    aclrtFree(outputRstdDeviceAddr);
    aclrtFree(outputXDeviceAddr);

    aclrtFree(outputYDeviceAddrBias);
    aclrtFree(outputMeanDeviceAddrBias);
    aclrtFree(outputRstdDeviceAddrBias);
    aclrtFree(outputXDeviceAddrBias);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    if (workspaceSizeBias > 0) {
        aclrtFree(workspaceAddrBias);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
