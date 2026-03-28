# aclnnCircularPad2dBackward

## 支持的产品型号
- 昇腾910B AI处理器。
- 昇腾910_93 AI处理器。

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnCircularPad2dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCircularPad2dBackward”接口执行计算。

- `aclnnStatus aclnnCircularPad2dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* padding, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnCircularPad2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

- 算子功能：circular_pad2d的反向传播, 前向计算参考[[aclnnCircularPad2d](aclnnCircularPad2d.md)]。
- 示例：

  ```
  输入gradOutput([[[1, 1, 1, 1, 1, 1, 1]]])
  self([[[1, 1, 1, 1, 1]]])
  padding([1, 1, 0, 0])
  输出为([[2, 1, 1, 1, 2]])
  ```

## aclnnCircularPad2dBackwardGetWorkspaceSize

- **参数说明：**

  - gradOutput（aclTensor*，计算输入）: 反向时输入的梯度数据，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32, 且数据类型与self一致，[数据格式](common/数据格式.md)支持ND，shape支持3-4维且维度需要与self和gradInput一致，shape需要与circular_pad2d正向传播的output一致。
  - self（aclTensor*，计算输入）：正向时待填充的原输入数据，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32, [数据格式](common/数据格式.md)支持ND，shape支持3-4维且维度需要与gradOutput和gradInput一致，shape与gradInput一致。
  - padding（aclIntArray*，计算输入）：正向时填充的维度，Device侧的aclIntArray数组，shape为1维，数据类型为INT64，[数据格式](common/数据格式.md)支持ND，长度为4，数值依次代表左右上下需要填充的值。padding前两个数值都需小于self最后一维度的大小，后两个数值需小于self倒数第二维度的大小。
  - gradInput（aclTensor*，计算输出）：反向时输出的梯度数据，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32, 且数据类型与self一致，[数据格式](common/数据格式.md)支持ND，shape支持3-4维且维度与gradOutput和self一致，shape与self一致。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现如下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. gradOutput, self, padding, gradInput任何一个为空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. gradOutput、self、padding和gradInput的数据类型或数据格式不在支持的范围之内。
                                        2. gradOutput、self、padding和gradInput的输入shape在支持范围之外。
                                        3. self为空tensor且存在非第一维度的大小为0。
                                        4. padding内的数值大于等于self的维度大小。
                                        5. gradOutput的shape需要与circular_pad2d正向传播的output一致。
  ```

## aclnnCircularPad2dBackward

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnCircularPad2dBackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制

- gradOutput的最后一维在不同类型下的大小需满足如下约束：
float16/bfloat16：(0， 16384)
float32：(0， 24576)

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_circular_pad2d_backward.h"
#include <iostream>
#include <vector>

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
    // 1. 固定写法，device/stream初始化, 参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> gradOutputShape = {1, 1, 4, 4};
    std::vector<int64_t> selfShape = {1, 1, 2, 2};
    std::vector<int64_t> gradInputShape = {1, 1, 2, 2};
    void* gradOutputDeviceAddr = nullptr;
    void* selfDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* self = nullptr;
    aclIntArray* padding = nullptr;
    aclTensor* gradInput = nullptr;
    std::vector<float> gradOutputHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> selfHostData = {1, 2, 3, 4};
    std::vector<int64_t> paddingData = {1, 1, 1, 1};
    std::vector<float> gradInputHostData = {0, 0, 0, 0};
    // 创建gradOutput aclTensor
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建padding aclIntArray
    padding = aclCreateIntArray(paddingData.data(), 4);
    CHECK_RET(padding != nullptr, return ret);
    // 创建gradInput aclTensor
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnCircularPad2dBackward第一段接口
    ret = aclnnCircularPad2dBackwardGetWorkspaceSize(gradOutput, self, padding, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCircularPad2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnCircularPad2dBackward第二段接口
    ret = aclnnCircularPad2dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCircularPad2dBackward failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(self);
    aclDestroyIntArray(padding);
    aclDestroyTensor(gradInput);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(selfDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
