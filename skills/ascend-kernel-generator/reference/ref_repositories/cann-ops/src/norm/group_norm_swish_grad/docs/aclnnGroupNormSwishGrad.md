# aclnnGroupNormSwishGrad

## 支持的产品型号

- Atlas A2 训练系列产品/Atlas 800I A2推理产品
- Atlas A3 训练系列产品/Atlas 800I A3推理产品

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnGroupNormSwishGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupNormSwishGrad”接口执行计算。

+ `aclnnStatus aclnnGroupNormSwishGradGetWorkspaceSize(const aclTensor* dy, const aclTensor* mean, const aclTensor* rstd, const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, int64_t group, char* dataFormat, float swishScale, bool dgammaIsRequire, bool dbetaIsRequire, aclTensor* dxOut, aclTensor* dgammaOutOptional, aclTensor* dbetaOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor)`
+ `aclnnStatus aclnnGroupNormSwishGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

GroupNormSwish的反向操作。

## aclnnGroupNormSwishGradGetWorkspaceSize

- **参数说明：**
  
  * dy (aclTensor\*, 计算输入)：输入张量，Device侧的aclTensor，反向计算的梯度，维度需大于一维，元素个数需要等于N\*C\*HxW，数据类型支持FLOAT32、FLOAT16、BFLOAT16，[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。

  * mean (aclTensor\*, 计算输入)：输入张量，Device侧的aclTensor，正向计算的第二个输出，表示input分组后每个组的均值，元素个数需要等于N\*group，数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据类型与$gamma$相同，其中`N`与$dy$的第0维度保持一致，[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。 

  * rstd (aclTensor\*, 计算输入)：输入张量，Device侧的aclTensor，正向计算的第三个输出，表示input分组后每个组的标准差倒数，元素个数需要等于N\*group，数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据类型与$gamma$相同，其中`N`与$dy$的第0维度保持一致，[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。

  * x (aclTensor\*, 计算输入)：输入张量，Device侧的aclTensor，正向的输入$x$，维度需大于一维，数据类型支持FLOAT32、FLOAT16、BFLOAT16，[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。

  * gamma (aclTensor\*, 计算输入)：输入张量，Device侧的aclTensor，表示每个channel的缩放系数，维度为一维，元素个数需要等于C，数据类型支持FLOAT32、FLOAT16、BFLOAT16，[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。

  * beta (aclTensor\*, 计算输入)：输入张量，Device侧的aclTensor，表示每个channel的偏移系数，维度为一维，元素个数需要等于C，数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据类型与$gamma$相同，[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。

  * numGroups (int64_t, 计算输入)：INT64常量，表示将输入gradOut的C维度分为group组，group需大于0，C必须可以被group整除并且比值不能超过4000。

  * dataFormat (char\*, 计算输入)：表示数据格式，建议值NCHW。

  * swishScale (float, 计算输入)：Swish计算公式中的系数，建议值1.0。

  * dgammaIsRequire (bool, 计算输入)：是否需要输出dgamma，建议值true。

  * dbetaIsRequire (bool, 计算输入)：是否需要输出dbeta，建议值true。

  * dxOut (aclTensor\*, 计算输出)：输出Tensor，Device侧的aclTensor，x的梯度，数据类型支持BFLOAT16、FLOAT16、FLOAT，数据类型和shape与$x$相同，[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。

  * dgammaOutOptional (aclTensor\*, 计算输出)：输出Tensor，Device侧的aclTensor，gamma的梯度，数据类型支持BFLOAT16、FLOAT16、FLOAT，数据类型和shape与$gamma$相同，[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。

  * dbetaOutOptional (aclTensor\*, 计算输出)：输出Tensor，Device侧的aclTensor，beta的梯度，数据类型支持BFLOAT16、FLOAT16、FLOAT，数据类型和shape与$gamma$相同，[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。

  * workspaceSize (uint64_t\*, 出参)：返回需要在Device侧申请的workspace大小。

  * executor (aclOpExecutor\**, 出参)：返回op执行器，包含算子计算流程。

- **返回值：**
  
  aclnnStatus: 返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
161001 ACLNN_ERR_PARAM_NULLPTR：1. 传入的dy、mean、rstd、x、gamma、beta、dx是空指针时。
161002 ACLNN_ERR_PARAM_INVALID：1. dy数据类型不在支持的范围之内。
                                2. mean、rstd、x、gamma、beta的数据类型与dy不同。
                                3. dxOut的数据类型与dy不同。
                                6. group不大于0。
                                7. C不能被group整除。
                                8. dy的元素个数不等于 N * C * HxW。
                                9. mean的元素个数不等于 N * group。
                                10. rstd的元素个数不等于 N * group。
                                11. x的元素个数不等于 N * C * HxW。
                                12. gamma的元素个数不等于 C。
                                13. beta的元素个数不等于 C。
                                14. C与group比值超过4000。
```

## aclnnGroupNormSwishGrad

- **参数说明：**
  
  * workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnGroupNormSwishGradGetWorkspaceSize获取。
  * executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制

无

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm_swish_grad.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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
  std::vector<float> dyHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> meanHostData = {2.0, 2};
  std::vector<float> rstdHostData = {2.0, 2};
  std::vector<float> xHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                  13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> gammaHostData = {2.0, 2, 2};
  std::vector<float> betaHostData = {2.0, 2, 2};
  std::vector<float> dxOutHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> dgammaOutHostData = {2.0, 2, 2};
  std::vector<float> dbetaOutHostData = {2.0, 2, 2};
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
```
