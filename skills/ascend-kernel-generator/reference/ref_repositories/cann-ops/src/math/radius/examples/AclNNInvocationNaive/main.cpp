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
#include "aclnn_radius.h"

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

#define ACL_TYPE aclDataType::ACL_INT32
constexpr int32_t  TYPE_SIZE = 4;

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
    std::vector<int64_t> inputXShape;
    std::vector<int64_t> inputYShape;
    std::vector<int64_t> inputPtrXShape;
    std::vector<int64_t> inputPtrYShape;
    std::vector<int64_t> outputShape;

    void *inputXDeviceAddr = nullptr;
    void *inputYDeviceAddr = nullptr;
    void *inputPtrXDeviceAddr = nullptr;
    void *inputPtrYDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;

    aclTensor *inputX = nullptr;
    aclTensor *inputY = nullptr;
    aclTensor *inputPtrX = nullptr;
    aclTensor *inputPtrY = nullptr;
    aclTensor *output = nullptr;

    size_t inputXShapeSize_1=1;
    size_t inputYShapeSize_1=1;
    size_t outputShapeSize_1=1;

    for(int i = 1; i < argc; i++){
        outputShape.push_back(std::atoi(argv[i]));
        outputShapeSize_1 *= std::atoi(argv[i]);
    }
    bool isEmpty = false;
    if(outputShapeSize_1 == 0){
        isEmpty = true;
        outputShape[1] = 1;
        outputShapeSize_1 = 2;
    }
    std::vector<int32_t> inputPtrXHostData = {};
    std::vector<int32_t> inputPtrYHostData = {};

    float r;
    int max_num_neighbors = 32;
    bool ignore_same_index = false;

    inputXShape.push_back(100);
    inputXShape.push_back(2);
    inputXShapeSize_1 = 200;

    inputYShape.push_back(50);
    inputYShape.push_back(2);
    inputYShapeSize_1 = 100;

    r = 1.0f;
    max_num_neighbors = 10;
    ignore_same_index = false;
    
    
    size_t dataType=TYPE_SIZE;
    std::vector<int32_t> inputXHostData(inputXShapeSize_1);
    std::vector<int32_t> inputYHostData(inputYShapeSize_1);
    std::vector<int32_t> outputHostData(outputShapeSize_1);
    

    size_t fileSize = 0;
    void** input1=(void**)(&inputXHostData);
    void** input2=(void**)(&inputYHostData);
    //读取数据
    ReadFile("../input/input_x.bin", fileSize, *input1, inputXShapeSize_1*dataType);
    ReadFile("../input/input_y.bin", fileSize, *input2, inputYShapeSize_1*dataType);

    INFO_LOG("Set input success");
    // 创建inputX aclTensor
    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, ACL_TYPE, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    // 创建inputY aclTensor
    ret = CreateAclTensor(inputYHostData, inputYShape, &inputYDeviceAddr, ACL_TYPE, &inputY);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    if(inputPtrXHostData.size()){
        ret = CreateAclTensor(inputPtrXHostData, inputPtrXShape, &inputPtrXDeviceAddr, aclDataType::ACL_INT32, &inputPtrX);
        CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    }
    if(inputPtrYHostData.size()){
        ret = CreateAclTensor(inputPtrYHostData, inputPtrYShape, &inputPtrYDeviceAddr, aclDataType::ACL_INT32, &inputPtrY);
        CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    }
    // 创建output aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, ACL_TYPE, &output);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);

    // 3. 调用CANN自定义算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 计算workspace大小并申请内存
    ret = aclnnRadiusGetWorkspaceSize(inputX, inputY, inputPtrX, inputPtrY, r, max_num_neighbors, ignore_same_index, output, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRadiusGetWorkspaceSize failed. ERROR: %d\n", ret); return FAILED);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
    }
    // 执行算子
    ret = aclnnRadius(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRadius failed. ERROR: %d\n", ret); return FAILED);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputShape);
    
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr,
                      size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
    void** output1=(void**)(&resultData);
    //写出数据
    if(isEmpty){
        outputShapeSize_1 = 0;
    }
    WriteFile("../output/output_y.bin", *output1, outputShapeSize_1*dataType);
    INFO_LOG("Write output success");

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(inputX);
    aclDestroyTensor(output);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(inputXDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return SUCCESS;
}