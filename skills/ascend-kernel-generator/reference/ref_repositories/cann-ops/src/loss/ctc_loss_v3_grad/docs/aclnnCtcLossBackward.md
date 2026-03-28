# aclnnCtcLossBackward 

## 支持的产品型号
- Atlas A2 训练系列产品

## 功能说明

算子功能：[aclnnCtcLoss](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/aclnnCtcLoss.md)的反向传播，计算CTC的损失梯度。

## 函数原型

每个算子分为[两段式接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用“aclnnCtcLossBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCtcLossBackward”接口执行计算。

- `aclnnStatus aclnnCtcLossBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclTensor* logProbs, const aclTensor* targets, const aclIntArray* inputLengths, const aclIntArray* targetLengths, const aclTensor* negLogLikelihood, const aclTensor* logAlpha, int64_t blank, bool zeroInfinity, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnCtcLossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnCtcLossBackwardGetWorkspaceSize

- **参数说明：**

  - gradOut(aclTensor*, 计算输入): 表示梯度更新系数，Device侧的aclTensor。数据类型支持FLOAT、DOUBLE，且数据类型必须和logProbs一致。shape为($N$)。支持[非连续的Tensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md)。[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - logProbs(aclTensor*, 计算输入): 表示输出的对数概率，Device侧的aclTensor。数据类型支持FLOAT、DOUBLE。shape为($T, N, C$)，$T$为输入长度，$N$为批处理大小，$C$为类别数，包括空白标识。支持[非连续的Tensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md)。[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - targets(aclTensor*, 计算输入): 表示包含目标序列的标签，Device侧的aclTensor。数据类型支持INT64、INT32、BOOL、FLOAT、FLOAT16数据类型。当shape为($N, S$)，$S$为不小于$targetLengths$中的最大值的值；或者shape为(SUM($targetLengths$))，假设$targets$是未填充的而且在1维内级联的。支持[非连续的Tensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md)。[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。数值必须小于$C$大于等于0。
  - inputLengths(aclIntArray*, 计算输入)：表示输入序列的实际长度，Host侧的aclIntArray。数组长度为$N$，数组中的每个值必须小于等于$T$。
  - targetLengths(aclIntArray*, 计算输入)：表示目标序列的实际长度，Host侧的aclIntArray。数组长度为$N$，当targets的shape为($N,S$)时，数组中的每个值必须小于等于$S$，最大值为$maxTargetLength$。
  - negLogLikelihood(aclTensor*, 计算输入)：表示相对于每个输入节点可微分的损失值，Device侧的aclTensor。数据类型支持FLOAT、DOUBLE，且数据类型必须和logProbs一致。shape为($N$)。支持[非连续的Tensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md)。[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - logAlpha(aclTensor*, 计算输入)：表示输入到目标的可能跟踪的概率，Device侧的aclTensor。数据类型支持FLOAT、DOUBLE，且数据类型必须和logProbs一致。shape必须为3维的非空Tensor, shape为($N, T, alpha$), 满足$(alpha - 1) / 2 >= maxTargetLength$ 。支持[非连续的Tensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md)。[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - blank(int, 计算输入)：表示空白标识，Host侧的整型。数值必须小于$C$大于等于0。
  - zeroInfinity(bool, 计算输入)：表示是否将无限损耗和相关梯度归零，Host侧的bool类型。
  - out(aclTensor*, 计算输出): 表示CTC的损失梯度，Device侧的aclTensor。数据类型支持FLOAT、DOUBLE，且数据类型必须和gradOut一致。shape为($T, N, C$)。支持[非连续的Tensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E9%9D%9E%E8%BF%9E%E7%BB%AD%E7%9A%84Tensor.md)。[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize(uint64_t*, 出参): 返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**, 出参): 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：传入的gradOut、logProbs、targets、inputLengths、targetLengths、negLogLikelihood、logAlpha、out是空指针时。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. gradOut、logProbs、targets、inputLengths、targetLengths、negLogLikelihood、logAlpha、out的数据类型不在支持的范围之内。
                                        2. gradOut、negLogLikelihood、logAlpha、out和logProbs数据类型不同。
                                        3. gradOut、logProbs、targets、negLogLikelihood、logAlpha、out的Tensor不满足对应的shape要求，或者inputLengths、targetLengths的ArrayList的长度不满足要求。
                                        4. blank不满足取值范围。
  ```

## aclnnCtcLossBackward

- **参数说明：**

  - workspace(void*, 入参): 在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnCtcLossBackwardGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参): op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参): 指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束说明
无

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E7%BC%96%E8%AF%91%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%A0%B7%E4%BE%8B_fuse.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ctc_loss_backward.h"

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
  // 1. （固定写法）device/stream初始化, 参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> gradOutShape = {4};
  // logProbsShape (T, N, C)
  std::vector<int64_t> logProbsShape = {12, 4, 5};
  std::vector<int64_t> targetsShape = {4, 7};
  std::vector<int64_t> negLoglikelihoodShape = {4};
  // logAlphaShape (N, T, X)  X = ((max(targetLengths) * 2 + 1) + 7) / 8 * 8;
  std::vector<int64_t> logAlphaShape = {4, 12, 16};
  std::vector<int64_t> outShape = {12, 4, 5};
  void* gradOutDeviceAddr = nullptr;
  void* logProbsDeviceAddr = nullptr;
  void* targetsDeviceAddr = nullptr;
  void* negLoglikelihoodDeviceAddr = nullptr;
  void* logAlphaDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* logProbs = nullptr;
  aclTensor* targets = nullptr;
  aclIntArray* inputLengths = nullptr;
  aclIntArray* targetLengths = nullptr;
  aclTensor* negLoglikelihood = nullptr;
  aclTensor* logAlpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> gradOutHostData = {1, 1, 1, 1};
  std::vector<float> logProbsHostData = {
    -1.0894, -2.7162, -0.9764, -1.9126, -2.6162,
    -2.0684, -2.4871, -2.0866, -1.7205, -0.7187,
    -2.4423, -1.2017, -1.4653, -1.1821, -2.5942,
    -2.4670, -2.7257, -1.4135, -2.1042, -0.7248,

    -3.7759, -1.3742, -1.2549, -1.5807, -1.4562,
    -1.3826, -1.8995, -1.8527, -0.9493, -2.8895,
    -1.6316, -2.6603, -2.5014, -0.6992, -1.8609,
    -1.9269, -2.2350, -0.8073, -1.8906, -1.8947,

    -0.3468, -2.5855, -2.0723, -2.7147, -3.6668,
    -0.9541, -1.7258, -2.0693, -1.6378, -2.1531,
    -3.5386, -3.4830, -0.2532, -2.0557, -3.3261,
    -1.1480, -1.8080, -0.8244, -3.2414, -3.1909,

    -0.8866, -0.7540, -4.4312, -3.4634, -2.6000,
    -1.2785, -1.8347, -3.3122, -0.7620, -2.8349,
    -1.4975, -1.3865, -0.9645, -3.8171, -2.0939,
    -2.3536, -2.0773, -1.4981, -0.8372, -2.0938,

    -1.2186, -0.8285, -2.9399, -2.1159, -2.3620,
    -2.3139, -0.6503, -2.7249, -1.2340, -3.7927,
    -0.7143, -2.5084, -3.2826, -2.6651, -1.1334,
    -1.6965, -1.9728, -2.3849, -1.6052, -0.9554,

    -1.6384, -1.2596, -2.1680, -1.8476, -1.3866,
    -3.0455, -0.5737, -2.5339, -2.1118, -1.6681,
    -2.4675, -2.8842, -0.4329, -3.6266, -1.6925,
    -3.1023, -2.7696, -1.2755, -0.6470, -2.4143,

    -2.0107, -2.0912, -1.3053, -0.8557, -3.0683,
    -1.2872, -3.6523, -1.6703, -2.7596, -0.8063,
    -2.4633, -1.2959, -1.6153, -2.3072, -1.0705,
    -3.0543, -0.6473, -1.1650, -2.9025, -2.7710,

    -3.5519, -2.0400, -1.8667, -1.4289, -0.8050,
    -1.4602, -0.7452, -1.5754, -3.1624, -3.1247,
    -1.4677, -1.2725, -2.9575, -1.8883, -1.2513,
    -1.2164, -1.5894, -2.2217, -2.3714, -1.2110,

    -2.0843, -0.6515, -1.4252, -2.9402, -2.7964,
    -1.5261, -2.5471, -1.7167, -1.9846, -0.9488,
    -1.4847, -1.7093, -1.4095, -1.7293, -1.7675,
    -0.9203, -4.2299, -1.8740, -1.4076, -1.6671,

    -1.9052, -0.8330, -2.1839, -2.2459, -1.6193,
    -2.9108, -1.2114, -1.4616, -1.7297, -1.4330,
    -2.2656, -0.7878, -1.8533, -1.8711, -2.0349,
    -2.2457, -2.1395, -1.4509, -0.7538, -2.6381,

    -0.8078, -2.1054, -2.6703, -1.1108, -3.3867,
    -1.7774, -1.8426, -1.9473, -1.3293, -1.3273,
    -1.3490, -1.9842, -2.5357, -2.2161, -0.8800,
    -1.5412, -1.8003, -2.7603, -0.8606, -2.0066,

    -1.8342, -2.2741, -1.8348, -1.5833, -0.9877,
    -3.5196, -2.3361, -0.9124, -0.9307, -2.5531,
    -1.4862, -1.2153, -1.4453, -3.4462, -1.5625,
    -2.6455, -1.4153, -1.3079, -1.1568, -2.2897};
  std::vector<int64_t> targetsHostData = {
    1, 2, 1, 1, 2, 4, 1,
    2, 2, 2, 2, 2, 2, 3,
    4, 2, 1, 4, 3, 1, 4,
    4, 1, 4, 2, 2, 2, 3};

  std::vector<float> negLoglikelihoodHostData = {10.1999, 16.1340, 14.9006,  9.3596};
  std::vector<float> logAlphaHostData = {
    -1.0894, -2.7162, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999, -99999,
    -4.8653, -2.2842, -6.4921, -3.9711, -99999, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -5.2121, -4.7967, -2.6162, -4.1742, -4.3179, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -6.0987, -5.0438, -3.3957, -6.7671, -4.4369, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -7.3173, -5.5735, -4.4384, -6.1313, -5.5627, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -8.9557, -6.6720, -5.7981, -6.1973, -6.7523, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -10.9664, -8.6661, -7.4600, -6.3671, -7.7544, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -14.5183, -10.6106, -10.7501, -7.8722, -9.6961, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -16.6026, -11.2422, -12.0691, -9.1833, -9.8069, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -18.5078, -12.0705, -12.7846, -11.1988, -10.6593, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,

    -2.0684, -2.0866, -99999, -99999, -99999, -99999, -99999,-99999, -99999, -99999, -99999, -99999, -99999, -99999,-99999, -99999,
    -3.4510, -3.2370, -3.4692, -99999, -99999, -99999, -99999,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -4.4051, -4.7144, -3.6073, -5.5385, -99999, -99999, -99999,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -5.6836, -7.1669, -4.6003, -6.7841, -6.8170, -99999, -99999,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -7.9975, -8.2040, -6.8402, -7.2185, -8.4212, -9.5419, -99999,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -11.0430, -9.9362, -9.6580, -8.8523, -10.0013, -10.6729, -12.5874,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -12.3302, -11.3209, -10.3815, -10.1533, -9.8642, -11.2589, -11.8226,-14.2577, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -13.7904, -12.5855, -11.5118, -11.1431, -10.7654, -11.2181, -12.2686,-13.3140, -15.7179, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -15.3165, -14.0400, -12.7439, -12.3341, -11.7695, -11.9899, -12.4443,-13.6840, -14.7536, -17.4346, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -18.2273, -15.2555, -15.4129, -13.2866, -14.2301, -12.6421, -14.4091,-13.6517, -16.2998, -16.1490, -20.3454, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,

    -2.4423, -2.5942, -99999, -99999, -99999, -99999, -99999,-99999, -99999, -99999, -99999, -99999, -99999, -99999,-99999, -99999,
    -4.0739, -3.6831, -4.2258, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -7.6125, -6.4925, -6.7635, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -9.1100, -8.3040, -7.4232, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -9.8243, -9.0682, -7.7908, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -12.2918, -10.3758, -10.0124, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -14.7551, -11.3089, -11.9478, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -16.2228, -12.5289, -12.3528, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -17.7075, -14.2718, -13.2285, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -19.9731, -16.2750, -15.1923, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,

    -2.4670, -0.7248, -99999, -99999, -99999, -99999, -99999,-99999, -99999, -99999, -99999, -99999, -99999, -99999,-99999, -99999,
    -4.3939, -2.4581, -2.6517, -2.9598, -99999, -99999, -99999,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -5.5419, -5.5142, -3.0051, -3.3784, -4.1078, -6.1507, -99999,-99999, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -7.8955, -6.9286, -5.2805, -4.5115, -5.3385, -5.0374, -8.5043,-7.6488, -99999, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -9.5920, -7.5617, -6.8010, -6.0444, -5.8452, -4.7597, -6.7031,-7.3228, -9.3453, -99999, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -12.6943, -9.8527, -9.5199, -8.2901, -8.3490, -6.6950, -7.7281,-5.8361, -10.3008, -10.6208, -99999, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -15.7486, -12.5670, -12.0336, -8.5306, -10.6803, -9.1337, -9.4448,-6.5472, -8.8790, -10.9199, -13.6751, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -16.9650, -13.7373, -12.7884, -10.0734, -9.6368, -9.2326, -9.8004,-8.6463, -7.6709, -10.9785, -12.0746, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -17.8853, -15.3655, -13.3814, -14.2154, -10.0586, -10.1583, -9.7039,-9.8935, -8.2713, -9.5090, -11.6105, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    -20.1310, -17.9262, -15.4983, -15.0687, -12.2888, -12.0440, -11.4581,-10.2538, -10.3368, -9.4675, -11.6393, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000, 0.0000};

  std::vector<float> outHostData = {
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
  };

  // 创建gradOut aclTensor
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建logProbs aclTensor
  ret = CreateAclTensor(logProbsHostData, logProbsShape, &logProbsDeviceAddr, aclDataType::ACL_FLOAT, &logProbs);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建targets aclTensor
  ret = CreateAclTensor(targetsHostData, targetsShape, &targetsDeviceAddr, aclDataType::ACL_INT64, &targets);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> inputLengthsSizeData = {10,10,10,10};
  inputLengths = aclCreateIntArray(inputLengthsSizeData.data(), 4);
  CHECK_RET(inputLengths != nullptr, return ACL_ERROR_BAD_ALLOC);
  std::vector<int64_t> targetLengthsSizeData = {2, 3, 1, 5};
  targetLengths = aclCreateIntArray(targetLengthsSizeData.data(), 4);
  CHECK_RET(targetLengths != nullptr, return ACL_ERROR_BAD_ALLOC);

  // 创建negLoglikelihood aclTensor
  ret = CreateAclTensor(negLoglikelihoodHostData, negLoglikelihoodShape, &negLoglikelihoodDeviceAddr, aclDataType::ACL_FLOAT, &negLoglikelihood);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建logAlpha aclTensor
  ret = CreateAclTensor(logAlphaHostData, logAlphaShape, &logAlphaDeviceAddr, aclDataType::ACL_FLOAT, &logAlpha);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnCtcLossBackward第一段接口
  ret = aclnnCtcLossBackwardGetWorkspaceSize(gradOut, logProbs, targets, inputLengths, targetLengths, negLoglikelihood, logAlpha, 0, false, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLossBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnCtcLossBackward第二段接口
  ret = aclnnCtcLossBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLossBackward failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的out值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和IntArray，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOut);
  aclDestroyTensor(logProbs);
  aclDestroyTensor(targets);
  aclDestroyIntArray(inputLengths);
  aclDestroyIntArray(targetLengths);
  aclDestroyTensor(negLoglikelihood);
  aclDestroyTensor(logAlpha);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(logProbsDeviceAddr);
  aclrtFree(targetsDeviceAddr);
  aclrtFree(negLoglikelihoodDeviceAddr);
  aclrtFree(logAlphaDeviceAddr);
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

