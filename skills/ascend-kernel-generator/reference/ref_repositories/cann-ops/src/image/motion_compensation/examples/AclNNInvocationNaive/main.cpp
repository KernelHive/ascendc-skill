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
#include <memory>

#include "acl/acl.h"
#include "aclnn_motion_compensation.h"

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

#define CHECK_FREE_RET(cond, return_expr)        \
    do {                                         \
        if (!(cond)) {                           \
            Finalize(deviceId, stream); \
            return_expr;                         \
        }                                        \
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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return SUCCESS;
}

int main(int argc, char **argv)
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    const int64_t N    = 1000;
    const int64_t ndim = 4;

    std::vector<float>    pointsHost(ndim * N);
    std::vector<uint64_t> timestampsHost(N);
    int64_t timestampMin, timestampMax;
    std::vector<float>    transMin(3), transMax(3);
    std::vector<float>    quatMin(4), quatMax(4);

    void *devPoints       = nullptr;
    void *devTimestamps   = nullptr;
    void *devOutPoints    = nullptr;

    std::vector<int64_t> pointsShape    = {ndim, N};
    std::vector<int64_t> timestampsShape= {N};
    std::vector<int64_t> scalarShape   = {1};
    std::vector<int64_t> scalar3Shape   = {3};
    std::vector<int64_t> scalar4Shape   = {4};

    size_t fileSize = 0;
    ReadFile("../input/input_points.bin",    sizeof(float) * ndim * N, pointsHost.data(),    sizeof(float) * ndim * N);
    ReadFile("../input/input_timestamps.bin", sizeof(uint64_t) * N,       timestampsHost.data(), sizeof(uint64_t) * N);
    ReadFile("../input/input_timestamp_min.bin", sizeof(int64_t),       &timestampMin,        sizeof(int64_t));
    ReadFile("../input/input_timestamp_max.bin", sizeof(int64_t),       &timestampMax,        sizeof(int64_t));
    ReadFile("../input/input_translation_min.bin", sizeof(float) * 3,    transMin.data(),      sizeof(float) * 3);
    ReadFile("../input/input_translation_max.bin", sizeof(float) * 3,    transMax.data(),      sizeof(float) * 3);
    ReadFile("../input/input_quaternion_min.bin",  sizeof(float) * 4,    quatMin.data(),       sizeof(float) * 4);
    ReadFile("../input/input_quaternion_max.bin",  sizeof(float) * 4,    quatMax.data(),       sizeof(float) * 4);
    INFO_LOG("Read input success");

    aclTensor *aclPoints       = nullptr;
    aclTensor *aclTimestamps   = nullptr;
    aclTensor *aclOutPoints    = nullptr;

    ret = CreateAclTensor(pointsHost,    pointsShape,    &devPoints,       ACL_FLOAT,  &aclPoints);
    ret |=    CreateAclTensor(timestampsHost,timestampsShape,&devTimestamps,   ACL_UINT64, &aclTimestamps);
    ret |=    CreateAclTensor(std::vector<float>(ndim * N),    pointsShape,    &devOutPoints,    ACL_FLOAT,  &aclOutPoints);
    
    CHECK_RET(ret == ACL_SUCCESS, "CreateAclTensor failed");

    aclFloatArray *aclTransMin = aclCreateFloatArray(transMin.data(), 3);
    std::unique_ptr<aclFloatArray, aclnnStatus (*)(const aclFloatArray *)> transMinPtr(aclTransMin, aclDestroyFloatArray);
    CHECK_RET(aclTransMin != nullptr, "CreateAclArray failed");

    aclFloatArray *aclTransMax = aclCreateFloatArray(transMax.data(), 3);
    std::unique_ptr<aclFloatArray, aclnnStatus (*)(const aclFloatArray *)> transMaxPtr(aclTransMax, aclDestroyFloatArray);
    CHECK_RET(aclTransMax != nullptr, "CreateAclArray failed");

    aclFloatArray *aclQuatMin = aclCreateFloatArray(quatMin.data(), 4);
    std::unique_ptr<aclFloatArray, aclnnStatus (*)(const aclFloatArray *)> quatMinPtr(aclQuatMin, aclDestroyFloatArray);
    CHECK_RET(aclQuatMin != nullptr, "CreateAclArray failed");

    aclFloatArray *aclQuatMax = aclCreateFloatArray(quatMax.data(), 4);
    std::unique_ptr<aclFloatArray, aclnnStatus (*)(const aclFloatArray *)> quatMaxPtr(aclQuatMax, aclDestroyFloatArray);
    CHECK_RET(aclQuatMax != nullptr, "CreateAclArray failed");

    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;

    aclnnStatus st = aclnnMotionCompensationGetWorkspaceSize(
        aclPoints, aclTimestamps, timestampMin, timestampMax,
        aclTransMin, aclTransMax, aclQuatMin, aclQuatMax,
        aclOutPoints, &workspaceSize, &executor);
    CHECK_RET(st == ACL_SUCCESS, "aclnnMotionCompensationGetWorkspaceSize failed %d");

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "Malloc workspace failed");
    }

    st = aclnnMotionCompensation(workspace, workspaceSize, executor, stream);
    CHECK_RET(st == ACL_SUCCESS, "aclnnMotionCompensation failed %d");

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

    const size_t outPointsSize = static_cast<size_t>(ndim * N * 4);

    std::vector<float>    outPointsHost(ndim * N);

    ret = aclrtMemcpy(outPointsHost.data(), outPointsSize,
                                devOutPoints, outPointsSize,
                                ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);

    WriteFile("../output/output_points.bin",    outPointsHost.data(), outPointsSize);
    INFO_LOG("Write output success");

    aclDestroyTensor(aclPoints);
    aclDestroyTensor(aclTimestamps);
    aclDestroyTensor(aclOutPoints);

    aclrtFree(devPoints);
    aclrtFree(devTimestamps);
    aclrtFree(devOutPoints);
    if (workspaceSize > 0) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return SUCCESS;
}
