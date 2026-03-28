# aclnnCircularPad2d

## 支持的产品型号
- Atlas 800I A2推理产品。
- Atlas A2训练系列产品。

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnCircularPad2dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCircularPad2d”接口执行计算。

- `aclnnStatus aclnnCircularPad2dGetWorkspaceSize(const aclTensor* self, const aclIntArray* padding, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnCircularPad2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

- 算子功能：使用输入循环填充输入tensor的最后两维。
- 示例：

```
输入tensor([[[[0,1,2],
              [3,4,5],
              [6,7,8]]]])
padding([2,2,2,2])
输出为([[[[4,5,3,4,5,3,4],
[7,8,6,7,8,6,7],
[1,2,0,1,2,0,1],
[4,5,3,4,5,3,4],
[7,8,6,7,8,6,7],
[1,2,0,1,2,0,1],
[4,5,3,4,5,3,4]]]])
```

## aclnnCircularPad2dGetWorkspaceSize

- **参数说明：**

  - self（aclTensor*, 计算输入）：待填充的原输入数据，Device侧的aclTensor。shape支持3-4维，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT8、INT32。[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。
  - padding（aclIntArray*, 计算输入）：输入中需要填充的维度，host侧的aclIntArray，shape为1维，数据类型为INT64，[数据格式](common/数据格式.md)支持ND，长度为4，数值依次代表左右上下需要填充的值。padding前两个数值都需小于self最后一维度的大小，后两个数值需小于self倒数第二维度的大小。
  - out（aclTensor*, 计算输出）：填充后的输出结果，Device侧的aclTensor。shape支持3-4维，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT8、INT32。[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)。out倒数第二维度的大小等于self倒数第二维度的大小加padding后两个数值，out最后一维度的大小等于self最后一维度的大小加padding前两个数值。
  - workspaceSize（uint64_t*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**, 出参）：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的self、padding、out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self、out的数据类型或数据格式不在支持的范围之内。
                                        2. self、out的数据类型不一致。
                                        3. self、padding和out的输入shape在支持范围之外。
                                        4. self为空tensor且存在非第一维度的大小为0。
                                        5. padding的数值大于等于self对应维度的大小。
                                        6. out后两维度的大小不等于self后两维度的大小加对应padding。
  ```

## aclnnCircularPad2d

- **参数说明：**

  - workspace（void*, 入参）：在Device侧申请的workspace内存地址。

  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnCircularPad2dGetWorkspaceSize获取。

  - executor（aclOpExecutor*, 入参）：op执行器，包含了算子计算流程。

  - stream（aclrtStream, 入参）：指定执行任务的AscendCL Stream流。


- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制

- out的最后一维在不同类型下的大小需满足如下约束：
int8：(0， 98304)
float16/bfloat16：(0， 49152)
int32/float32：(0， 24576)
