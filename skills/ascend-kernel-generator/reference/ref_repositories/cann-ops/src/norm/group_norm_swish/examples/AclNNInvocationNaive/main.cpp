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
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>


#include "acl/acl.h"
#include "aclnn_group_norm_swish.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

bool ReadFile(const std::string &filePath, size_t fileSize, void *buffer,
              size_t bufferSize) {
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

bool WriteFile(const std::string &filePath, const void *buffer, size_t size) {
  if (buffer == nullptr) {
    ERROR_LOG("Write file failed. buffer is nullptr");
    return false;
  }

  int fd =
      open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
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

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，acl初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return FAILED);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return FAILED);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return FAILED);

  return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return FAILED);

  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return FAILED);

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0,
                            aclFormat::ACL_FORMAT_ND, shape.data(),
                            shape.size(), *deviceAddr);
  return SUCCESS;
}

int main(int argc, char **argv) {
  // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return FAILED);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> inputXShape = {100, 32};
  std::vector<int64_t> inputGammaShape = {32};
  std::vector<int64_t> inputBetaShape = {32};
  std::vector<int64_t> outputOutShape = {100, 32};
  std::vector<int64_t> outputMeanOutShape = {100, 8};
  std::vector<int64_t> outputRstdOutShape = {100, 8};
  void *inputXDeviceAddr = nullptr;
  void *inputGammaDeviceAddr = nullptr;
  void *inputBetaDeviceAddr = nullptr;
  void *outputOutDeviceAddr = nullptr;
  void *outputMeanDeviceAddr = nullptr;
  void *outputRstdDeviceAddr = nullptr;
  aclTensor *inputX = nullptr;
  aclTensor *inputGamma = nullptr;
  aclTensor *inputBeta = nullptr;
  aclTensor *outputOut = nullptr;
  aclTensor *outputMean = nullptr;
  aclTensor *outputRstd = nullptr;
  size_t inputXShapeSize = GetShapeSize(inputXShape);
  size_t inputGammaShapeSize = GetShapeSize(inputGammaShape);
  size_t inputBetaShapeSize = GetShapeSize(inputBetaShape);
  size_t outputOutShapeSize = GetShapeSize(outputOutShape);
  size_t outputMeanOutShapeSize = GetShapeSize(outputMeanOutShape);
  size_t outputRstdOutShapeSize = GetShapeSize(outputRstdOutShape);
  std::vector<aclFloat16> inputXHostData(inputXShapeSize);
  std::vector<aclFloat16> inputGammaHostData(inputGammaShapeSize);
  std::vector<aclFloat16> inputBetaHostData(inputBetaShapeSize);
  std::vector<aclFloat16> outputOutHostData(outputOutShapeSize);
  std::vector<aclFloat16> outputMeanOutHostData(outputMeanOutShapeSize);
  std::vector<aclFloat16> outputRstdOutHostData(outputRstdOutShapeSize);
  size_t dataType = sizeof(uint16_t);
  size_t fileSize = 0;
  void **input1 = (void **)(&inputXHostData);
  void **input2 = (void **)(&inputGammaHostData);
  void **input3 = (void **)(&inputBetaHostData);
  // 读取数据
  ReadFile("../input/input_x.bin", fileSize, *input1,
           inputXShapeSize * dataType);
  ReadFile("../input/input_gamma.bin", fileSize, *input2,
           inputGammaShapeSize * dataType);
  ReadFile("../input/input_beta.bin", fileSize, *input3,
           inputBetaShapeSize * dataType);

  INFO_LOG("Set input success");
  // 创建input aclTensor
  ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr,
                        aclDataType::ACL_FLOAT16, &inputX);
  CHECK_RET(ret == ACL_SUCCESS, return FAILED);
  ret = CreateAclTensor(inputGammaHostData, inputGammaShape,
                        &inputGammaDeviceAddr, aclDataType::ACL_FLOAT16,
                        &inputGamma);
  CHECK_RET(ret == ACL_SUCCESS, return FAILED);
  ret = CreateAclTensor(inputBetaHostData, inputBetaShape, &inputBetaDeviceAddr,
                        aclDataType::ACL_FLOAT16, &inputBeta);
  CHECK_RET(ret == ACL_SUCCESS, return FAILED);
  // 创建output aclTensor
  ret = CreateAclTensor(outputOutHostData, outputOutShape, &outputOutDeviceAddr,
                        aclDataType::ACL_FLOAT16, &outputOut);
  CHECK_RET(ret == ACL_SUCCESS, return FAILED);
  ret = CreateAclTensor(outputMeanOutHostData, outputMeanOutShape,
                        &outputMeanDeviceAddr, aclDataType::ACL_FLOAT16,
                        &outputMean);
  CHECK_RET(ret == ACL_SUCCESS, return FAILED);
  ret = CreateAclTensor(outputRstdOutHostData, outputRstdOutShape,
                        &outputRstdDeviceAddr, aclDataType::ACL_FLOAT16,
                        &outputRstd);
  CHECK_RET(ret == ACL_SUCCESS, return FAILED);

  // 3. 调用CANN自定义算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;
  // 计算workspace大小并申请内存
  int64_t numGroups = 8;
  char *dataFormatOptional = "NCHW";
  double eps = 0.00001;
  bool activateSwish = true;
  double scale = 1.0;
  ret = aclnnGroupNormSwishGetWorkspaceSize(
      inputX, inputGamma, inputBeta, numGroups, dataFormatOptional, eps,
      activateSwish, scale, outputOut, outputMean, outputRstd, &workspaceSize,
      &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnGroupNormSwishGetWorkspaceSize failed. ERROR: %d\n", ret);
      return FAILED);
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return FAILED;);
  }
  // 执行算子
  ret = aclnnGroupNormSwish(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnGroupNormSwish failed. ERROR: %d\n", ret);
            return FAILED);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return FAILED);

  // 5.
  // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  std::vector<aclFloat16> resultData(outputOutShapeSize, 0);
  std::vector<aclFloat16> resultMeanData(outputMeanOutShapeSize, 0);
  std::vector<aclFloat16> resultRstdData(outputRstdOutShapeSize, 0);
  ret =
      aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                  outputOutDeviceAddr, outputOutShapeSize * sizeof(aclFloat16),
                  ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return FAILED);
  ret = aclrtMemcpy(
      resultMeanData.data(), resultMeanData.size() * sizeof(resultMeanData[0]),
      outputMeanDeviceAddr, outputMeanOutShapeSize * sizeof(aclFloat16),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy resultMeanData from device to host failed. ERROR: %d\n",
                ret);
      return FAILED);
  ret = aclrtMemcpy(
      resultRstdData.data(), resultRstdData.size() * sizeof(resultRstdData[0]),
      outputRstdDeviceAddr, outputRstdOutShapeSize * sizeof(aclFloat16),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy resultRstdData from device to host failed. ERROR: %d\n",
                ret);
      return FAILED);
  void **output1 = (void **)(&resultData);
  void **output2 = (void **)(&resultMeanData);
  void **output3 = (void **)(&resultRstdData);
  // 写出数据
  WriteFile("../output/output_out.bin", *output1,
            outputOutShapeSize * dataType);
  WriteFile("../output/output_mean.bin", *output2,
            outputMeanOutShapeSize * dataType);
  WriteFile("../output/output_rstd.bin", *output3,
            outputRstdOutShapeSize * dataType);
  INFO_LOG("Write output success");

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(inputX);
  aclDestroyTensor(inputGamma);
  aclDestroyTensor(inputBeta);
  aclDestroyTensor(outputOut);
  aclDestroyTensor(outputMean);
  aclDestroyTensor(outputMean);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(inputXDeviceAddr);
  aclrtFree(inputGammaDeviceAddr);
  aclrtFree(inputBetaDeviceAddr);
  aclrtFree(outputOutDeviceAddr);
  aclrtFree(outputMeanDeviceAddr);
  aclrtFree(outputRstdDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return SUCCESS;
}