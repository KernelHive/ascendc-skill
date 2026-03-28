# aclnnMaskedSelect

## 支持的产品型号
- Atlas 推理系列产品
- Atlas 训练系列产品
- Atlas A2 训练系列产品/Atlas A3 训练系列产品

## 接口原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnMaskedSelectGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMaskedSelect”接口执行计算。

- `aclnnStatus aclnnMaskedSelectGetWorkspaceSize(const aclTensor* self, const aclTensor* mask, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnMaskedSelect(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

算子功能：根据一个布尔掩码张量（mask）中的值选择输入张量（self）中的元素作为输出，形成一个新的一维张量。

## aclnnMaskedSelectGetWorkspaceSize

- **参数说明：**

  - self (aclTensor*, 计算输入)：功能描述中的输入张量`self`，Device侧的aclTensor。shape需要与mask满足[broadcast关系](common/broadcast关系.md)。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - Atlas 推理系列产品/Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT32、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品：数据类型支持BFLOAT16、FLOAT16、FLOAT32、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
  - mask (aclTensor*, 计算输入)：功能描述中的布尔掩码张量`mask`，Device侧的aclTensor。shape要和self满足[broadcast关系](common/broadcast关系.md)。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。数据类型支持UINT8、BOOL。
  - out (aclTensor*, 计算输出)：功能描述中的输出一维张量，Device侧的aclTensor。shape为一维，且元素个数为mask和self广播后的shapesize。不支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - Atlas 推理系列产品/Atlas 训练系列产品：数据类型支持FLOAT16、FLOAT32、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品：数据类型支持BFLOAT16、FLOAT16、FLOAT32、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL。
  - workspaceSize (uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor (aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self、mask、out是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. self和mask的数据类型不在支持的范围之内。
                                       2. self和mask的shape无法做broadcast。
                                       3. out的shape不是一维时。
                                       4. out的元素个数不等于self和mask广播后的shapesize时。
  ```

## aclnnMaskedSelect

- **参数说明：**

  - workspace(void*, 入参) ：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnMaskedSelectGetWorkspaceSize获取。

  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制

无

## 算子原型
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">MaskedSelectV3</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <!-- op::DataType::DT_FLOAT,   op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_DOUBLE, op::DataType::DT_BOOL, op::DataType::DT_BF1 -->
<tr><td align="center">x</td><td align="center">-</td><td align="center">float, int32, int64, float16, int16, int8, uint8, double, bool, bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">mask</td><td align="center">-</td><td align="center">bool</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">1D</td><td align="center">float, int32, int64, float16, int16, int8, uint8, double, bool, bfloat16</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">masked_select_v3</td></tr>
</table>

## 调用示例

详见[MaskedSelectV3自定义算子样例说明算子调用章节](../README.md#算子调用)