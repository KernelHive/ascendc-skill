声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# aclnnDiag

## 支持的产品型号

- Atlas 训练系列产品
- Atlas A2 训练系列产品/Atlas A3 训练系列产品

## 接口原型
每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnDiagGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDiag”接口执行计算。

- `aclnnStatus aclnnDiagGetWorkspaceSize(const aclTensor* self, int64_t diagonal, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnDiag(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能描述

- 算子功能：

  如果输入是向量(一维向量)，则返回二维矩阵张量，其中input元素为对角线;
  如果输入是二维张量，则输出一维向量，取值为diagonal指定的输入矩阵的对角线元素。


## aclnnDiagGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入)：self最大维度不能超过2。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。输入为空tensor时，输出类型不能是复数类型COMPLEX64。
    - Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX64。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX64、BFLOAT16。

  - diagonal(int64_t, 计算输入)：对应逻辑表达中的对角线输入，数据类型支持INT64。

  - out(aclTensor*, 计算输出)：输出张量，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX64。
    - Atlas A2 训练系列产品/Atlas A3 训练系列产品：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX64、BFLOAT16。

  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  
  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self或out是空指针。
  161002 (ACLNN_ERR_PARAM_INVALID): 1. self和out的数据类型不在支持的范围之内。
                                    2. diagonal不在支持的数据类型范围之内。
                                    3. out的shape与实际输出shape不匹配。
  ```

## aclnnDiag

- **参数说明：**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnDiagGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束与限制
无

## 算子原型
<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">DiagV2</th></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">默认值</td></tr>
<tr><td align="center">x</td><td align="center">-</td><td align="center">int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float16, bfloat16, double, bool, complex32, complex128, complex64</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">diagonal</td><td align="center">-</td><td align="center">int64</td><td align="center">\</td><td align="center">\</td></tr>

<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">-</td><td align="center">int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float16, bfloat16, double, bool, complex32, complex128, complex64</td><td align="center">ND</td><td align="center">\</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">diag_v2</td></td></tr>
</table>

## 调用示例

详见[DiagV2自定义算子样例说明算子调用章节](../README.md#算子调用)