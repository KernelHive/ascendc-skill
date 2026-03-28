# aclRfft1D 

## 支持的产品型号 

* Atlas A2 训练系列产品/Atlas A3 训练系列产品

## 接口原型 
每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclRfft1DGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclRfft1D”接口执行计算。

* aclnnStatus aclRfft1DGetWorkspaceSize(const aclTensor* self, int64_t n, int64_t dim, int64_t norm, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
* aclnnStatus aclRfft1D(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)

## 功能描述 
* 算子功能：对输入张量self进行RFFT（傅里叶变换）计算，输出是一个包含非负频率的复数张量。
* 计算公式：

  $$
  {\displaystyle X_{k}=\sum _{n=0}^{N-1}x_{n}\cdot e^{-i2\pi {\tfrac {k}{N}}n}}
  $$
  
* 示例：
  假设self为 {1, 2, 3, 4} 则out = {10, 0, -2, 2, -2, 0} = {10 + 0j, -2 + 2j, -2 + 0j} （自定义）

## aclRfft1DGetWorkspaceSize 

* 参数说明：
    * self(aclTensor\*, 计算输入): 即公式中的输入。数据类型支持FLOAT，[数据格式](common/数据格式.md)支持ND。
    * n(int64_t, 计算输入): 表示信号长度。数据类型支持INT64。如果给定，则在计算Rfft1D之前，输入将被补零或修剪到此长度。n取值范围为[1, 4096]，2的n次幂最大值为262144。
    * dim(int64_t, 计算输入): 表示维度。数据类型支持INT64。如果给定，则RFFT将应用于指定的维度。支持的值为-1。
    * norm(int64_t, 计算输入): 表示归一化模式。数据类型支持INT64。支持取值：1表示不归一化，2表示按1/n归一化，3表示按1/sqrt(n)归一化。
    * out (aclTensor\*, 计算输出): 表示公式中的输出。数据类型支持FLOAT。[数据格式](common/数据格式.md)支持ND。
    * workspaceSize (uint64_t, 出参): 返回需要在Device侧申请的workspace大小。
    * executor(aclOpExecutor\*, 出参): 返回op执行器，包含了算子计算流程。


* 返回值：

    aclnnStatus：返回状态码，具体具体参见[aclnn返回码](common/aclnn返回码.md)。
    ```
    第一段接口完成入参校验，出现以下场景时报错：
    返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 输入或输出tensor为空。
    返回161002（ACLNN_ERR_PARAM_INVALID）: 1. self的数据类型不在支持的范围内。
    返回561103（ACLNN_ERR_INNER_NULLPTR）: 1. 中间结果为null。
    返回561101（ACLNN_ERR_INNER_CREATE_EXECUTOR）: 1. 执行者为null。                              
    ```

## aclRfft1D 
* 参数说明：
    * workspace（void*, 入参）：在Device侧申请的workspace内存地址。
    * workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclRfft1DGetWorkspaceSize获取。
    * executor（aclOpExecutor*, 入参）：op执行器，包含了算子计算流程。
    * stream（aclrtStream, 入参）：指定执行任务的AscendCL Stream流。
* 返回值：

    aclnnStatus：返回状态码，具体具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制 

无

## 调用示例 

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/acl_rfft1d.h"
#define CHECK_RET(cond, return_expr)   \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)
#define LOG_PRINT(message, ...)       \
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
int Init(int32_t deviceId, aclrtStream* stream) {
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

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {

    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> selfShape = {1, 1, 8};
    std::vector<int64_t> outShape = {1, 1, 5, 2};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    int n = 8;
    int dim = -1;
    int norm = 1; // backward - 1, forward - 2, ortho - 3

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclRfft1DGetWorkspaceSize(self, n, dim, norm, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclRfft1DGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclRfft1D(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclRfft1D failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    aclDestroyTensor(self);
    aclDestroyTensor(out);

    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```