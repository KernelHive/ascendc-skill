# aclStft
## 支持的产品型号

- 昇腾910B AI处理器。
- 昇腾910_93 AI处理器。

## 接口原型

每个算子分为两段式接口，必须先调用“aclStftGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclStft”接口执行计算。

* `aclnnStatus aclStftGetWorkspaceSize(const aclTensor *self, const aclTensor *windowOptional, aclTensor *out, int64_t nFft, int64_t hopLength, int64_t winLength, bool normalized, bool onesided, bool returnComplex, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclStft(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

- **算子功能**: 计算输入在滑动窗口内的傅里叶变换。
- **计算公式**:

    $$
    X[w,m]=\sum_{k=0}^{winLength-1}window[k]*self[m*hopLength+k]*exp(-j*\frac{2{\pi}wk}{nFft})
    $$
    
   其中，$w$为FFT的频点；$m$为滑动窗口的index；$self$为1维或2维tensor，当$self$是1维时，其为一个时序采样序列，当$self$是2维时，其为多个时序采样序列；$hopLength$为滑动窗口大小；$window$为1维tensor，是STFT的窗函数（例如hann_window），其长度为$winLength$；$exp(-j*\frac{2{\pi}wk}{nFft})$为旋转因子。

## aclStftGetWorkspaceSize

-   **参数说明：**
    -   self（aclTensor\*，计算输入）：待计算的输入，要求是一个1D/2D的Tensor，shape为\(L\)/\(B, L\)，其中，L为时序采样序列的长度，B为时序采样序列的个数。数据类型支持FLOAT32、DOUBLE、COMPLEX64、COMPLEX128，支持非连续的Tensor，数据格式要求为ND。
    -   windowOptional（aclTensor\*，计算输入）：要求是一个1D的Tensor，shape为\(winLength\)，winLength为STFT窗函数的长度。数据类型支持FLOAT32、DOUBLE、COMPLEX64、COMPLEX128，且数据类型与self保持一致，数据格式要求为ND。
    -   out（aclTensor\*，计算输出）：self在window内的傅里叶变换结果，要求是一个2D/3D/4D的Tensor，数据类型支持FLOAT32、DOUBLE、COMPLEX64、COMPLEX128，支持非连续的Tensor，数据格式要求为ND。
        - 如果returnComplex=True，out是shape为\[N, T\]或者\[B, N, T\]的复数tensor。
        - 如果returnComplex=False，out是shape为\[N, T, 2\]或者\[B, N, T, 2\]的实数tensor。
        
        其中，N=nFft\(onesided=False\)或者(nFft // 2 + 1)(onesided=True);
        T是滑动窗口的个数，T = (L - nFft) // hopLength + 1。
    -   nFft（int64\_t，计算输入）：Host侧的int，FFT的点数（大于0）。
    -   hopLength（int64\_t，计算输入）：Host侧的int，滑动窗口的间隔（大于0）。
    -   winLength（int64\_t，计算输入）：Host侧的int，window的大小（大于0）。
    -   normalized（bool，计算输入）：Host侧的bool，是否对傅里叶变换结果进行标准化。
    -   onesided（bool，计算输入）：Host侧的bool，是否返回全部的结果或者一半结果。
    -   returnComplex（bool，计算输入）：Host侧的bool，确认返回值是complex tensor或者是实部、虚部分开的tensor。
    -   workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
    -   executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。


-   **返回值：**

    返回aclnnStatus状态码，具体参见aclnn返回码。

    ```
    第一段接口完成入参校验，出现以下场景时报错：
    返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self、out是空指针
    返回161002（ACLNN_ERR_PARAM_INVALID）：1. self的格式不在支持的范围之内；
                                         2. self、windowOptional、out的数据类型不在平台的支持范围之内；
                                         3. nFft、hopLength、winLength输入无效值；
                                         4. self、windowOptional、out的维度不在支持的范围之内;
    ```

## aclStft

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclStftGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

## 约束与限制

-   输入self与PyTorch接口的不同：PyTorch接口的输入self为原始输入；aclStftGetWorkspaceSize的入参self是原始输入经过前端PyTorch补pad后得到的结果；
-   PyTorch接口调用STFT时，self数据类型仅支持FLOAT32、DOUBLE；
-   nFft <= L；
-   winLength <= nFft;
-   当normalized=True时，
    
    $$
    STFT(w,m)=\frac{1}{\sqrt{N}}X[w,m]
    $$
-  self、windowOptional、returnComplex、out输入和输出数据类型的对应关系如下
  
  |self|windowOptional|returnComplex|out|
  |-------|-------|-------|-------|
  |FLOAT32| FLOAT32        | True          | COMPLEX64  |
  | DOUBLE     | DOUBLE         | True          | COMPLEX128 |
  |COMPLEX64|COMPLEX64|True|COMPLEX64|
  |COMPLEX128|COMPLEX128|True|COMPLEX128|
  |FLOAT32| FLOAT32        | False     | FLOAT32 |
  | DOUBLE     | DOUBLE         | False     | DOUBLE |
  |COMPLEX64|COMPLEX64|False|FLOAT32|
  |COMPLEX128|COMPLEX128|False|DOUBLE|


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/acl_stft.h"

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
  // 1. （固定写法）device/stream初始化，参考acl对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {5};
  std::vector<int64_t> windowShape = {4};
  std::vector<int64_t> outShape = {3, 1, 2};
  void* selfDeviceAddr = nullptr;
  void* windowDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* window = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {1, 6, 8, 5, 7};
  std::vector<float> windowHostData = {1, 1, 1, 1};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  // 创建window aclTensor
  ret = CreateAclTensor(windowHostData, windowShape, &windowDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int n_fft = 4;
  int hop_length = 2;
  int win_length = 4;
  bool normalized = false;
  bool onesided = true;
  bool returnComplex = false;
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclStft第一段接口
  ret = aclStftGetWorkspaceSize(self, window, out, n_fft, hop_length, win_length, normalized, onesided, returnComplex, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclStftGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclStft第二段接口
  ret = aclStft(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclStft failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(window);
  aclDestroyTensor(out);
  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(windowDeviceAddr);
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

