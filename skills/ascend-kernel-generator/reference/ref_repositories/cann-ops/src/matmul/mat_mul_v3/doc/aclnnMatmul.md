# aclnnMatmul

## 支持的产品型号

- Atlas 推理系列产品。
- Atlas 训练系列产品。
- Atlas A2训练系列产品/Atlas 800I A2推理产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnMatmulGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMatmul”接口执行计算。

- `aclnnStatus aclnnMatmulGetWorkspaceSize(const aclTensor *self, const aclTensor *mat2, aclTensor *out, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor)`

- `aclnnStatus aclnnMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- 算子功能：完成张量self与张量mat2的矩阵乘计算（支持1维到6维作为输入的矩阵乘）。
  相似接口有aclnnMm（支持2维Tensor作为输入的矩阵乘）和aclnnBatchMatmul（仅支持3维的矩阵乘，其中第1维为batch）。
- 计算公式：

  $$
  result=self @ mat2
  $$

## aclnnMatmulGetWorkspaceSize
- **参数说明：**

  - self（aclTensor*，计算输入）：表示矩阵乘的第一个矩阵，公式中的self，Device侧aclTensor。数据类型需要与mat2满足数据类型推导规则（参见[互推导关系](./互推导关系.md)和[约束与限制](#约束与限制)）。数据格式支持ND。shape维度支持1维到6维，并且需要与mat2满足broadcast关系。支持非连续的Tensor。
    - Atlas 训练系列AI处理器、Atlas 推理系列 AI处理器：数据类型支持FLOAT16、FLOAT32。
    - Atlas A2训练/Atlas 800I A2推理系列 AI处理器：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
  - mat2（aclTensor*，计算输入）：表示矩阵乘的第二个矩阵，公式中的mat2，Device侧的aclTensor，数据类型需要与self满足数据类型推导规则（参见[互推导关系](./互推导关系.md)和[约束与限制](#约束与限制)）。数据格式支持ND。shape维度支持1维到6维，并且需要与self满足broadcast关系。支持非连续的Tensor。mat2的Reduce维度需要与self的Reduce维度大小相等。
    - Atlas 训练系列 AI处理器、Atlas 推理系列 AI处理器：数据类型支持FLOAT16、FLOAT32。
    - Atlas A2训练/Atlas 800I A2推理系列 AI处理器：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
  - out（aclTensor*，计算输出）：表示矩阵乘的输出矩阵，公式中的out，Device侧aclTensor。数据类型需要与self与mat2推导之后的数据类型保持一致（参见[互推导关系](./互推导关系.md)和[约束与限制](#约束与限制)）。数据格式支持ND。shape维度支持1维到6维。
    - Atlas 训练系列 AI处理器、Atlas 推理系列 AI处理器：数据类型支持FLOAT16、FLOAT32。
    - Atlas A2训练/Atlas 800I A2推理系列 AI处理器：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
  - cubeMathType（INT8，计算输入）：用于指定Cube单元的计算逻辑，Host侧的整型。数据类型支持INT8。注意：如果输入的数据类型存在互推导关系，该参数默认对互推导后的数据类型进行处理。支持的枚举值如下：
    * 0：KEEP_DTYPE，保持输入的数据类型进行计算。
      * Atlas 训练系列 AI处理器、Atlas 推理系列 AI处理器：当输入数据类型为FLOAT32时不支持该选项。
    * 1：ALLOW_FP32_DOWN_PRECISION，支持将输入数据降精度计算。
      * Atlas 训练系列 AI处理器、Atlas 推理系列 AI处理器：当输入数据类型为FLOAT32时，会转换为FLOAT16计算。当输入为其他数据类型时不做处理。
      * Atlas A2训练/Atlas 800I A2推理系列 AI处理器：当输入数据类型为FLOAT32时，会转换为HFLOAT32计算。当输入为其他数据类型时不做处理。
    * 2：USE_FP16，支持将输入降精度至FLOAT16计算。
      * Atlas A2训练/Atlas 800I A2推理系列 AI处理器：当输入数据类型为BFLOAT16时不支持该选项。
    * 3：USE_HF32，支持将输入降精度至数据类型HFLOAT32计算。
      * Atlas 训练系列 AI处理器、Atlas 推理系列 AI处理器：不支持该选项。
      * Atlas A2训练/Atlas 800I A2推理系列 AI处理器：当输入数据类型为FLOAT32时，会转换为HFLOAT32计算。当输入为其他数据类型时不支持该选项。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码。

```
第一段接口完成入参校验，出现以下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、mat2或out是空指针。
161002(ACLNN_ERR_PARAM_INVALID): 1. self和mat2的数据类型和数据格式不在支持的范围之内。
                                 2. self和mat2无法做数据类型推导。
                                 3. 推导出的数据类型无法转换为指定输出out的类型。
```


## aclnnMatmul

- **参数说明：**

- workspace(void*, 入参)：在Device侧申请的workspace内存地址。
- workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnMatmulGetWorkspaceSize获取。
- executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
- stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus：返回状态码。

## 约束与限制
- Atlas A2训练/Atlas 800I A2推理系列 AI处理器：不支持两个输入分别为BFLOAT16和FLOAT16的数据类型推导。不支持两个输入分别为BFLOAT16和FLOAT32的数据类型推导。
- self和mat2都是1维时，cubeMathType不生效。

## 调用示例

详见[MatMulV3自定义算子样例说明算子调用章节](../README.md#算子调用)

