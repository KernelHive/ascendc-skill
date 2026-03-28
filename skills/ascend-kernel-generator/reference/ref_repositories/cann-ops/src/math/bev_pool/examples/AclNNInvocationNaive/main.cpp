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
#include "aclnn_bev_pool.h"

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

    size_t B = 2, N = 4, D = 6, fH = 8, fW = 10;
    size_t D_Z = 4, D_Y = 6, D_X = 8, C = 12;
    size_t N_points = 20, N_pillar = 10;

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> bev_feat_shapeData = {B, D_Z, D_Y, D_X, C};
    aclIntArray* bev_feat_shape = aclCreateIntArray(bev_feat_shapeData.data(), bev_feat_shapeData.size());

    std::vector<int64_t> depthShape = {B, N, D, fH, fW};
    std::vector<int64_t> featShape = {B, N, fH, fW, C};
    std::vector<int64_t> ranks_depthShape = {N_points};
    std::vector<int64_t> ranks_featShape = {N_points};
    std::vector<int64_t> ranks_bevShape = {N_points};
    std::vector<int64_t> interval_startsShape = {N_pillar};
    std::vector<int64_t> interval_lengthsShape = {N_pillar};
    std::vector<int64_t> outShape = {B*C*D_Z*D_Y*D_X};

    void *depthDeviceAddr = nullptr;
    void *featDeviceAddr = nullptr;
    void *ranks_depthDeviceAddr = nullptr;
    void *ranks_featDeviceAddr = nullptr;
    void *ranks_bevDeviceAddr = nullptr;
    void *interval_startsDeviceAddr = nullptr;
    void *interval_lengthsDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;

    aclTensor *depth = nullptr;
    aclTensor *feat = nullptr;
    aclTensor *ranks_depth = nullptr;
    aclTensor *ranks_feat = nullptr;
    aclTensor *ranks_bev = nullptr;
    aclTensor *interval_starts = nullptr;
    aclTensor *interval_lengths = nullptr;
    aclTensor* out = nullptr;

    std::vector<aclFloat16> depthHostData(B*N*D*fH*fW);
    std::vector<aclFloat16> featHostData(B*N*fH*fW*C);
    std::vector<int32_t> ranks_depthHostData(N_points);
    std::vector<int32_t> ranks_featHostData(N_points);
    std::vector<int32_t> ranks_bevHostData(N_points);
    std::vector<int32_t> interval_startsHostData(N_pillar);
    std::vector<int32_t> interval_lengthsHostData(N_pillar);
    std::vector<aclFloat16> outHostData(B*C*D_Z*D_Y*D_X, aclFloatToFloat16(0.0));


    size_t dataType = 2;
    size_t dataType1 = 4;
    size_t fileSize = 0;
    void** input1=(void**)(&depthHostData);
    void** input2=(void**)(&featHostData);
    void** input3=(void**)(&ranks_depthHostData);
    void** input4=(void**)(&ranks_featHostData);
    void** input5=(void**)(&ranks_bevHostData);
    void** input6=(void**)(&interval_startsHostData);
    void** input7=(void**)(&interval_lengthsHostData);
    //读取数据
    ReadFile("../input/input_depth.bin", fileSize, *input1, B * N * D * fH * fW * dataType);
    ReadFile("../input/input_feat.bin", fileSize, *input2, B * N * fH * fW * C * dataType);
    ReadFile("../input/input_ranks_depth.bin", fileSize, *input3, N_points * dataType1);
    ReadFile("../input/input_ranks_feat.bin", fileSize, *input4, N_points * dataType1);
    ReadFile("../input/input_ranks_bev.bin", fileSize, *input5, N_points * dataType1);
    ReadFile("../input/input_interval_starts.bin", fileSize, *input6, N_pillar * dataType1);
    ReadFile("../input/input_interval_lengths.bin", fileSize, *input7, N_pillar * dataType1);

    INFO_LOG("Set input success");
    // 创建input aclTensor
    ret = CreateAclTensor(depthHostData, depthShape, &depthDeviceAddr, aclDataType::ACL_FLOAT16, &depth);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    ret = CreateAclTensor(featHostData, featShape, &featDeviceAddr, aclDataType::ACL_FLOAT16, &feat);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    ret = CreateAclTensor(ranks_depthHostData, ranks_depthShape, &ranks_depthDeviceAddr, aclDataType::ACL_INT32, &ranks_depth);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    ret = CreateAclTensor(ranks_featHostData, ranks_featShape, &ranks_featDeviceAddr, aclDataType::ACL_INT32, &ranks_feat);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    ret = CreateAclTensor(ranks_bevHostData, ranks_bevShape, &ranks_bevDeviceAddr, aclDataType::ACL_INT32, &ranks_bev);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    ret = CreateAclTensor(interval_startsHostData, interval_startsShape, &interval_startsDeviceAddr, aclDataType::ACL_INT32, &interval_starts);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    ret = CreateAclTensor(interval_lengthsHostData, interval_lengthsShape, &interval_lengthsDeviceAddr, aclDataType::ACL_INT32, &interval_lengths);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建output aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    // 3. 调用CANN自定义算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 计算workspace大小并申请内存
    ret = aclnnBevPoolGetWorkspaceSize(depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, bev_feat_shape, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBevPoolGetWorkspaceSize failed. ERROR: %d\n", ret); return FAILED);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
    }
    // 执行算子
    ret = aclnnBevPool(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBevPool failed. ERROR: %d\n", ret); return FAILED);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<aclFloat16> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
    void ** output1=(void **)(&resultData);
    //写出数据
    WriteFile("../output/output.bin", *output1, B*C*D_Z*D_Y*D_X * dataType);
    INFO_LOG("Write output success");

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(depth);
    aclDestroyTensor(feat);
    aclDestroyTensor(ranks_depth);
    aclDestroyTensor(ranks_feat);
    aclDestroyTensor(ranks_bev);
    aclDestroyTensor(interval_starts);
    aclDestroyTensor(interval_lengths);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(depthDeviceAddr);
    aclrtFree(featDeviceAddr);
    aclrtFree(ranks_depthDeviceAddr);
    aclrtFree(ranks_featDeviceAddr);
    aclrtFree(ranks_bevDeviceAddr);
    aclrtFree(interval_startsDeviceAddr);
    aclrtFree(interval_lengthsDeviceAddr);
    aclrtFree(outDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return SUCCESS;
}
