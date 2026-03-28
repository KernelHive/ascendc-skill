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

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_bidirection_lstm.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
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
  // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int time_step = 2;
  int batch_size = 32;
  int input_size = 32;
  int hidden_size = 32;

  int64_t numLayers = 1;
  bool isbias = true;
  bool batchFirst = false;
  bool bidirection = true;

  std::vector<int64_t> selfShape = {time_step, batch_size, input_size};
  std::vector<int64_t> weightHIShape = {4 * hidden_size, input_size};
  std::vector<int64_t> weightHHShape = {4 * hidden_size, hidden_size};
  std::vector<int64_t> initHShape = {2, batch_size, hidden_size};
  std::vector<int64_t> initCShape = {2, batch_size, hidden_size};
  std::vector<int64_t> biasHIShape = {4 * hidden_size};
  std::vector<int64_t> biasHHShape = {4 * hidden_size};  
  std::vector<int64_t> outShape = {time_step, batch_size, 2 * hidden_size};
  std::vector<int64_t> outHShape = {2, batch_size, hidden_size};
  std::vector<int64_t> outCShape = {2, batch_size, hidden_size};

  void* selfDeviceAddr = nullptr;
  void* weightHIDeviceAddr = nullptr;
  void* weightHHDeviceAddr = nullptr;
  void* weightHIReverseDeviceAddr = nullptr;
  void* weightHHReverseDeviceAddr = nullptr;
  void* initHDeviceAddr = nullptr;
  void* initCDeviceAddr = nullptr;
  void* biasHIDeviceAddr = nullptr;
  void* biasHHDeviceAddr = nullptr;
  void* biasHIReverseDeviceAddr = nullptr;
  void* biasHHReverseDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* outHDeviceAddr = nullptr;
  void* outCDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* weightHI = nullptr;
  aclTensor* weightHH = nullptr;
  aclTensor* weightHIReverse = nullptr;
  aclTensor* weightHHReverse = nullptr;
  aclTensor* biasHI = nullptr;
  aclTensor* biasHH = nullptr;
  aclTensor* biasHIReverse = nullptr;
  aclTensor* biasHHReverse = nullptr;
  aclTensor* initH = nullptr;
  aclTensor* initC = nullptr;
  aclTensor* out = nullptr;
  aclTensor* outH = nullptr;
  aclTensor* outC = nullptr;

  std::vector<uint16_t> selfHostData(GetShapeSize(selfShape));
  std::vector<uint16_t> weightHIHostData(GetShapeSize(weightHIShape));
  std::vector<uint16_t> weightHHHostData(GetShapeSize(weightHHShape));
  std::vector<uint16_t> biasHIHostData(GetShapeSize(biasHIShape));
  std::vector<uint16_t> biasHHHostData(GetShapeSize(biasHHShape));
  std::vector<uint16_t> initHHostData(GetShapeSize(initHShape));
  std::vector<uint16_t> initCHostData(GetShapeSize(initCShape));
  std::vector<uint16_t> outHostData(GetShapeSize(outShape));
  std::vector<uint16_t> outHHostData(GetShapeSize(outHShape));
  std::vector<uint16_t> outCHostData(GetShapeSize(outCShape));

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHIHostData, weightHIShape, &weightHIDeviceAddr, aclDataType::ACL_FLOAT16, &weightHI);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHHHostData, weightHHShape, &weightHHDeviceAddr, aclDataType::ACL_FLOAT16, &weightHH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(initHHostData, initHShape, &initHDeviceAddr, aclDataType::ACL_FLOAT16, &initH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);  
  ret = CreateAclTensor(initCHostData, initCShape, &initCDeviceAddr, aclDataType::ACL_FLOAT16, &initC);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHIHostData, biasHIShape, &biasHIDeviceAddr, aclDataType::ACL_FLOAT16, &biasHI);
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  ret = CreateAclTensor(biasHHHostData, biasHHShape, &biasHHDeviceAddr, aclDataType::ACL_FLOAT16, &biasHH);
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  ret = CreateAclTensor(weightHIHostData, weightHIShape, &weightHIReverseDeviceAddr, aclDataType::ACL_FLOAT16, &weightHIReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHHHostData, weightHHShape, &weightHHReverseDeviceAddr, aclDataType::ACL_FLOAT16, &weightHHReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHIHostData, biasHIShape, &biasHIReverseDeviceAddr, aclDataType::ACL_FLOAT16, &biasHIReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  ret = CreateAclTensor(biasHHHostData, biasHHShape, &biasHHReverseDeviceAddr, aclDataType::ACL_FLOAT16, &biasHHReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);  
  ret = CreateAclTensor(outHHostData, outHShape, &outHDeviceAddr, aclDataType::ACL_FLOAT16, &outH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outCHostData, outCShape, &outCDeviceAddr, aclDataType::ACL_FLOAT16, &outC);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnBidirectionLSTM第一段接口
  ret = aclnnBidirectionLSTMGetWorkspaceSize(self, initH, initC, weightHI, weightHH,
                                            biasHI, biasHH, weightHIReverse, weightHHReverse, biasHIReverse, biasHHReverse,
                                            numLayers, isbias, batchFirst, bidirection,
                                            out, outH, outC,
                                            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBidirectionLSTMGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnBidirectionLSTM第二段接口
  ret = aclnnBidirectionLSTM(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBidirectionLSTM failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outDeviceAddr);
  PrintOutResult(outHShape, &outHDeviceAddr);
  PrintOutResult(outCShape, &outCDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(weightHI);
  aclDestroyTensor(weightHH);
  aclDestroyTensor(initH);
  aclDestroyTensor(initC);
  aclDestroyTensor(biasHI);
  aclDestroyTensor(biasHH);
  aclDestroyTensor(weightHIReverse);
  aclDestroyTensor(weightHHReverse);
  aclDestroyTensor(biasHIReverse);
  aclDestroyTensor(biasHHReverse);
  aclDestroyTensor(out);
  aclDestroyTensor(outH);
  aclDestroyTensor(outC);

  // 7. 释放device资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(weightHIDeviceAddr);
  aclrtFree(weightHHDeviceAddr);
  aclrtFree(initHDeviceAddr);
  aclrtFree(initCDeviceAddr);
  aclrtFree(biasHIDeviceAddr);
  aclrtFree(biasHHDeviceAddr);
  aclrtFree(weightHIReverseDeviceAddr);
  aclrtFree(weightHHReverseDeviceAddr);
  aclrtFree(biasHIReverseDeviceAddr);
  aclrtFree(biasHHReverseDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(outHDeviceAddr);
  aclrtFree(outCDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}