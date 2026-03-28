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
#include "aclnn_points_to_voxel.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) (void)fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) (void)fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) (void)fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

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
    std::streamsize bytesRead = buf->sgetn(static_cast<char *>(buffer), size);
    if (bytesRead != size) {
        ERROR_LOG("read file size error");
    }
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
    std::vector<float> voxel_size = {1.3, 1.5, 1.6};
    std::vector<float> coors_range = {1.1, 1.3, 1.4, 98.4, 87.6, 103.7};
    int32_t max_points = 5;
    bool reverse_index = false;
    int32_t max_voxels = 500;
    aclFloatArray* voxel_sizeAcl = aclCreateFloatArray(voxel_size.data(), voxel_size.size());
    aclFloatArray* coors_rangeAcl = aclCreateFloatArray(coors_range.data(), coors_range.size());
    std::vector<int64_t> inputPointsShape = {4, 1000};
    std::vector<int64_t> outputVoxelsShape = {500, 5, 4};
    std::vector<int64_t> outputCoorsShape = {500, 3};
    std::vector<int64_t> outputNumPointsPerVoxelShape = {500};
    std::vector<int64_t> outputvoxelNumShape = {1};
    void *inputPointsDeviceAddr = nullptr;
    void *outputVoxelsDeviceAddr = nullptr;
    void *outputCoorsDeviceAddr = nullptr;
    void *outputNumPointsPerVoxelDeviceAddr = nullptr;
    void *outputvoxelNumDeviceAddr = nullptr;
    aclTensor *inputPoints = nullptr;
    aclTensor *outputVoxels = nullptr;
    aclTensor *outputCoors = nullptr;
    aclTensor *outputNumPointsPerVoxel = nullptr;
    aclTensor *outputvoxelNum = nullptr;
    size_t inputPointsShapeSize_1 = inputPointsShape[0] * inputPointsShape[1];
    size_t outputVoxelsShapeSize_1 = outputVoxelsShape[0] * outputVoxelsShape[1] * outputVoxelsShape[2];
    size_t outputCoorsShapeSize_1 = outputCoorsShape[0] * outputCoorsShape[1];
    size_t outputNumPointsPerVoxelSize_1 = outputNumPointsPerVoxelShape[0];
    size_t outputvoxelNumSize_1 = outputvoxelNumShape[0];
    size_t dataType = 4;
    std::vector<float> inputPointsHostData(inputPointsShape[0] * inputPointsShape[1]);
    std::vector<float> outputVoxelsHostData(outputVoxelsShape[0] * outputVoxelsShape[1] * outputVoxelsShape[2]);
    std::vector<int32_t> outputCoorsHostData(outputCoorsShape[0] * outputCoorsShape[1]);
    std::vector<int32_t> outputNumPointsPerVoxelHostData(outputNumPointsPerVoxelShape[0]);
    std::vector<int32_t> outputvoxelNumData(outputvoxelNumShape[0]);

    size_t fileSize = 0;
    void** input1 = (void**)(&inputPointsHostData);

    //读取数据
    ReadFile("../input/input_points.bin", fileSize, *input1, inputPointsShapeSize_1 * dataType);

    INFO_LOG("Set input success");
    // 创建input aclTensor
    ret = CreateAclTensor(inputPointsHostData, inputPointsShape, &inputPointsDeviceAddr, aclDataType::ACL_FLOAT, &inputPoints);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
   
    // 创建output aclTensor
    ret = CreateAclTensor(outputVoxelsHostData, outputVoxelsShape, &outputVoxelsDeviceAddr, aclDataType::ACL_FLOAT, &outputVoxels);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    ret = CreateAclTensor(outputCoorsHostData, outputCoorsShape, &outputCoorsDeviceAddr, aclDataType::ACL_INT32, &outputCoors);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    ret = CreateAclTensor(outputNumPointsPerVoxelHostData, outputNumPointsPerVoxelShape, &outputNumPointsPerVoxelDeviceAddr, aclDataType::ACL_INT32, &outputNumPointsPerVoxel);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    ret = CreateAclTensor(outputvoxelNumData, outputvoxelNumShape, &outputvoxelNumDeviceAddr, aclDataType::ACL_INT32, &outputvoxelNum);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);            
    // 3. 调用CANN自定义算子库API

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 计算workspace大小并申请内存
    ret = aclnnPointsToVoxelGetWorkspaceSize(inputPoints,  
                                             voxel_sizeAcl, coors_rangeAcl, max_points, reverse_index, max_voxels,
                                             outputVoxels, outputCoors, outputNumPointsPerVoxel, outputvoxelNum, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPointsToVoxelGetWorkspaceSize failed. ERROR: %d\n", ret); return FAILED);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
    }
    // 执行算子
    ret = aclnnPointsToVoxel(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPointsToVoxel failed. ERROR: %d\n", ret); return FAILED);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size1 = GetShapeSize(outputVoxelsShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), outputVoxelsDeviceAddr,
                    size1 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
    void** output1 = (void**)(&resultData1);
    //写出数据
    WriteFile("../output/output_voxels.bin", *output1, outputVoxelsShapeSize_1 * dataType);
    INFO_LOG("Write output1 success");

    auto size2 = GetShapeSize(outputCoorsShape);
    std::vector<int32_t> resultData2(size2, 0);
    ret = aclrtMemcpy(resultData2.data(), resultData2.size() * sizeof(resultData2[0]), outputCoorsDeviceAddr,
                    size2 * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
    void** output2 = (void**)(&resultData2);
    //写出数据
    WriteFile("../output/output_coors.bin", *output2, outputCoorsShapeSize_1 * dataType);
    INFO_LOG("Write output2 success");

    auto size3 = GetShapeSize(outputNumPointsPerVoxelShape);
    std::vector<int32_t> resultData3(size3, 0);
    ret = aclrtMemcpy(resultData3.data(), resultData3.size() * sizeof(resultData3[0]), outputNumPointsPerVoxelDeviceAddr,
                    size3 * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
    void** output3 = (void**)(&resultData3);
    //写出数据
    WriteFile("../output/output_num_points_per_voxel.bin", *output3, outputNumPointsPerVoxelSize_1 * dataType);
    INFO_LOG("Write output3 success");

    auto size4 = GetShapeSize(outputvoxelNumShape);
    std::vector<int32_t> resultData4(size4, 0);
    ret = aclrtMemcpy(resultData4.data(), resultData4.size() * sizeof(resultData4[0]), outputvoxelNumDeviceAddr,
                    size4 * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
    void** output4 = (void**)(&resultData4);
    //写出数据
    WriteFile("../output/output_voxel_num.bin", *output4, outputvoxelNumSize_1 * dataType);
    INFO_LOG("Write output4 success");

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(inputPoints);
    aclDestroyTensor(outputVoxels);
    aclDestroyTensor(outputCoors);
    aclDestroyTensor(outputNumPointsPerVoxel);
    aclDestroyTensor(outputvoxelNum);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(inputPointsDeviceAddr);
    aclrtFree(outputVoxelsDeviceAddr);
    aclrtFree(outputCoorsDeviceAddr);
    aclrtFree(outputNumPointsPerVoxelDeviceAddr);
    aclrtFree(outputvoxelNumDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return SUCCESS;
}