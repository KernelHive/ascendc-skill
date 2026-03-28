# aclnnDeepNorm

## 支持的产品型号

- Atlas 推理系列产品。
- Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnDeepNormGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDeepNorm”接口执行计算。

- `aclnnStatus aclnnDeepNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gx, const aclTensor *beta, const aclTensor *gamma, double alphaOptional, double epsilonOptional, const aclTensor *meanOut, const aclTensor *rstdOut, const aclTensor *yOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnDeepNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

-  算子功能：实现DeepNorm算子功能。
-  计算公式：

   $$
   DeepNorm(x_i^{\prime}) = (\frac{x_i^{\prime} - \bar{x^{\prime}}}{rstd}) * gamma + beta,
   $$

   $$
   \text { where } rstd = \sqrt{\frac{1}{n} \sum_{i=1}^n (x^{\prime}_i - \bar{x^{\prime}})^2 + eps} , \quad \operatorname{x^{\prime}_i} = alpha * x_i + gx_i
   $$

## aclnnDeepNormGetWorkspaceSize

-   **参数说明**：
    * x（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入`x`，shape支持2-8维度，数据格式支持ND。不支持空tensor。
      * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
      * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    * gx（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入`gx`，shape支持2-8维度，shape维度和输入x的维度相同，数据格式支持ND。不支持空tensor。
      * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
      * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    * beta（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入`beta`，shape支持1-7维度，shape维度和输入x后几维的维度相同，后几维表示需要norm的维度，数据格式支持ND。不支持空tensor。
      * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
      * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    * gamma（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入`gamma`，shape支持1-7维度，shape维度和输入x后几维的维度相同，后几维表示需要norm的维度，数据格式支持ND。不支持空tensor。
      * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
      * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    * alphaOptional（double，计算输入）：Host侧的double，公式中的输入`alpha`，是输入x的权重。
    * epsilonOptional（double，计算输入）：Host侧的double，公式中的输入`eps`，用于防止除0错误。
    * meanOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出`mean`，数据类型支持FLOAT32，shape支持2-8维度，shape与输入x满足broadcast关系（前几维的维度和输入x前几维的维度相同，前几维表示不需要norm的维度，其余维度大小为1），数据格式支持ND。不支持空tensor。
    * rstdOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出`rstd`，数据类型支持FLOAT32，shape支持2-8维度，shape与输入x满足broadcast关系（前几维的维度和输入x前几维的维度相同，前几维表示不需要norm的维度，其余维度大小为1），数据格式支持ND。不支持空tensor。
    * yOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出`y`，shape支持2-8维度，shape维度和输入x的维度相同，数据格式支持ND。不支持空tensor。
      * Atlas 推理系列产品：数据类型支持FLOAT32、FLOAT16。
      * Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    * workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
    * executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

-   **返回值**

    aclnnStatus：返回状态码。

    ```
    第一段接口完成入参校验，出现以下场景时报错：
    返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针。
    返回161002（ACLNN_ERR_PARAM_INVALID）：1. 输入和输出的数据类型不在支持的范围之内。
                                          2. 输入和输出的shape不匹配或者不在支持的维度范围内。
    ```

## aclnnDeepNorm

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnDeepNormGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    aclnnStatus：返回状态码。

## 约束与限制

-  功能维度：
    -  数据类型支持：
        - Atlas 推理系列产品：x、gx、beta、gamma、y支持FLOAT32、FLOAT16。
        - Atlas A2 训练系列产品/Atlas A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：x、gx、beta、gamma、y支持FLOAT32、FLOAT16、BFLOAT16。
        - rstd、mean支持：FLOAT32。

    -  数据格式支持：ND。

-  未支持类型说明：
    -  DOUBLE：指令不支持DOUBLE。
    -  是否支持空tensor：不支持空进空出。
    -  是否支持非连续tensor：不支持输入非连续。

-  边界值场景说明：
    -  当输入是inf时，输出为inf。
    -  当输入是nan时，输出为nan。
