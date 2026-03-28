# Log

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 800I A2推理产品


## 功能描述

- 算子功能：对输入张量中的每个元素`x`计算`log`函数，并将结果返回到输出张量。

- 计算公式：

  $$
  y=log(x * scale + shift)
  $$

## 实现原理

Log算子通过依次调用`Ascend C`的`API`接口：`Muls`、`Log`、`Muls`，分步计算实现Log函数计算。

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnLogGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLog”接口执行计算。

* `aclnnStatus aclnnLogGetWorkspaceSize(const aclTensor *x, const aclTensor *base, const aclTensor *scale, , const aclTensor *shift,const aclTensor *out, uint64_t workspaceSize, aclOpExecutor **executor)`;
* `aclnnStatus aclnnLog(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子。

### aclnnLogGetWorkspaceSize

- **参数说明：**

  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x，数据类型支持BFLOAT16、FLOAT16、FLOAT32，数据格式支持ND。
  - base（false\*，计算输入）：必选参数，Device侧的false，数据类型支持flase。
  - scale（false\*，计算输入）：必选参数，Device侧的false，数据类型支持false。
  - shift（false\*，计算输入）：必选参数，Device侧的false，数据类型支持false。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出y，数据类型支持BFLOAT16、FLOAT16、FLOAT32，数据格式支持ND，输出维度与x一致。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。



- **返回值：**
  
  返回aclnnStatus状态码。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x、base、scale、shift的数据类型和数据格式不在支持的范围内。
    ```

### aclnnLog

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnFFNGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码。

## 约束与限制

- x的数据类型支持BFLOAT16、FLOAT16、FLOAT32，数据格式只支持ND

## 算子原型

<table border="1">
  <tr>
    <th align="center" rowspan="2">算子类型(OpType)</th>
    <th colspan="4" align="center">Log</th>
  </tr>
  <tr>
    <th align="center">name</th>
    <th align="center">Type</th>
    <th align="center">data type</th>
    <th align="center">format</th>
  </tr>
  <tr>
    <td rowspan="1" align="center">算子输入</td>
    <td align="center">x</td>
    <td align="center">tensor</td>
    <td align="center">bfloat16,float32,float16</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td rowspan="3" align="center">算子属性</td>
    <td align="center">base</td>
    <td align="center">float</td>
    <td align="center">float</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td align="center">scale</td>
    <td align="center">float</td>
    <td align="center">float</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td align="center">shift</td>
    <td align="center">float</td>
    <td align="center">float</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td>
    <td align="center">tensor</td>
    <td align="center">bfloat16,float32,float16</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td align="center">核函数名</td>
    <td colspan="4" align="center">log</td>
  </tr>
</table>

## 调用示例

详见[Log自定义算子样例说明算子调用章节](../README.md#算子调用)