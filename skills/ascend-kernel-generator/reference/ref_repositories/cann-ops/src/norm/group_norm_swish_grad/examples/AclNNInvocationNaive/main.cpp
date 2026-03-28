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
#include "aclnnop/aclnn_group_norm_swish_grad.h"

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define CHECK_RET(cond, return_expr) \
    do {                               \
    if (!(cond)) {                   \
        return_expr;                   \
    }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
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

int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> dyShape = {2, 3, 4};
    std::vector<int64_t> meanShape = {2, 1};
    std::vector<int64_t> rstdShape = {2, 1};
    std::vector<int64_t> xShape = {2, 3, 4};
    std::vector<int64_t> gammaShape = {3};
    std::vector<int64_t> betaShape = {3};
    std::vector<int64_t> dxOutShape = {2, 3, 4};
    std::vector<int64_t> dgammaOutShape = {3};
    std::vector<int64_t> dbetaOutShape = {3};
    void* dyDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* dxOutDeviceAddr = nullptr;
    void* dgammaOutDeviceAddr = nullptr;
    void* dbetaOutDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* dxOut = nullptr;
    aclTensor* dgammaOut = nullptr;
    aclTensor* dbetaOut = nullptr;
    size_t dataType = 8; 


    size_t dyShapeSize = GetShapeSize(dyShape);
    size_t meanShapeSize = GetShapeSize(meanShape);
    size_t rstdShapeSize = GetShapeSize(rstdShape);
    size_t xShapeSize = GetShapeSize(xShape);
    size_t gammaShapeSize = GetShapeSize(gammaShape);
    size_t betaShapeSize = GetShapeSize(betaShape);
    size_t dxOutShapeSize = GetShapeSize(dxOutShape);
    size_t dgammaOutShapeSize = GetShapeSize(dgammaOutShape);
    size_t dbetaOutShapeSize = GetShapeSize(dbetaOutShape);

    std::vector<float> dyHostData(dyShapeSize);
    std::vector<float> meanHostData(meanShapeSize);
    std::vector<float> rstdHostData(rstdShapeSize);
    std::vector<float> xHostData(xShapeSize);
    std::vector<float> gammaHostData(gammaShapeSize);
    std::vector<float> betaHostData(betaShapeSize);
    std::vector<float> dxOutHostData(dxOutShapeSize);
    std::vector<float> dgammaOutHostData(dgammaOutShapeSize);
    std::vector<float> dbetaOutHostData(dbetaOutShapeSize);

    size_t fileSize = 0;
    void** input1=(void**)(&dyHostData);
    void** input2=(void**)(&meanHostData);
    void** input3=(void**)(&rstdHostData);
    void** input4=(void**)(&xHostData);
    void** input5=(void**)(&gammaHostData);
    void** input6=(void**)(&betaHostData);


    //读取数据
    ReadFile("../input/dy.bin", fileSize, *input1, dyShapeSize * dataType);
    ReadFile("../input/mean.bin", fileSize, *input2, meanShapeSize * dataType);
    ReadFile("../input/rstd.bin", fileSize, *input3, rstdShapeSize * dataType);
    ReadFile("../input/x.bin", fileSize, *input4, xShapeSize * dataType);
    ReadFile("../input/gamma.bin", fileSize, *input5, gammaShapeSize * dataType);
    ReadFile("../input/beta.bin", fileSize, *input6, betaShapeSize * dataType);
    INFO_LOG("Set input success");
    
    int64_t group = 1;
    char* dataFormat = nullptr;
    float swishScale = 1.0f;
    bool dgammaIsRequire = true;
    bool dbetaIsRequire = true;
    // 创建dy aclTensor
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mean aclTensor
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rstd aclTensor
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gamma aclTensor
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建beta aclTensor
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dxOut aclTensor
    ret = CreateAclTensor(dxOutHostData, dxOutShape, &dxOutDeviceAddr, aclDataType::ACL_FLOAT, &dxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dgammaOut aclTensor
    ret = CreateAclTensor(dgammaOutHostData, dgammaOutShape, &dgammaOutDeviceAddr, aclDataType::ACL_FLOAT, &dgammaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dbetaOut aclTensor
    ret = CreateAclTensor(dbetaOutHostData, dbetaOutShape, &dbetaOutDeviceAddr, aclDataType::ACL_FLOAT, &dbetaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnGroupNormSwishGrad第一段接口
    ret = aclnnGroupNormSwishGradGetWorkspaceSize(dy, mean, rstd, x, gamma, beta, group, dataFormat, swishScale, dgammaIsRequire, dbetaIsRequire, dxOut, dgammaOut, dbetaOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwishGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnGroupNormSwishGrad第二段接口
    ret = aclnnGroupNormSwishGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwishGrad failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(dxOutShape);
    ret = aclrtMemcpy(dxOutHostData.data(), dxOutHostData.size() * sizeof(dxOutHostData[0]), dxOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("dxOutHostData[%ld] is: %f\n", i, dxOutHostData[i]);
    }

    size = GetShapeSize(dgammaOutShape);
    ret = aclrtMemcpy(dgammaOutHostData.data(), dgammaOutHostData.size() * sizeof(dgammaOutHostData[0]), dgammaOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("dgammaOutHostData[%ld] is: %f\n", i, dgammaOutHostData[i]);
    }

    size = GetShapeSize(dbetaOutShape);
    ret = aclrtMemcpy(dbetaOutHostData.data(), dbetaOutHostData.size() * sizeof(dbetaOutHostData[0]), dbetaOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("dbetaOutHostData[%ld] is: %f\n", i, dbetaOutHostData[i]);
    }
    //写出数据
    void ** output1=(void **)(&dxOutHostData);
    void ** output2=(void **)(&dgammaOutHostData);
    void ** output3=(void **)(&dbetaOutHostData);
    dataType = 4;
    WriteFile("../output/npu_dx.bin", *output1, dxOutShapeSize * dataType);
    WriteFile("../output/npu_dgamma.bin", *output2, dgammaOutShapeSize * dataType);
    WriteFile("../output/npu_dbeta.bin", *output3, dbetaOutShapeSize * dataType);
    INFO_LOG("Write output success");
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(dy);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(dxOut);
    aclDestroyTensor(dgammaOut);
    aclDestroyTensor(dbetaOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(dyDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(dxOutDeviceAddr);
    aclrtFree(dgammaOutDeviceAddr);
    aclrtFree(dbetaOutDeviceAddr);

    if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
