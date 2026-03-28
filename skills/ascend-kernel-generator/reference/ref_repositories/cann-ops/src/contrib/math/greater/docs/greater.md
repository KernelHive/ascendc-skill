# Greater

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 800I A2推理产品

## 功能描述

- 算子功能：对两个张量 `x1`，`x2` 逐元素比较大小，`greater` 判断 `x1` 中的每个元素是否大于 `x2` 中的对应元素，并将比较结果（布尔值）返回到输出张量 `y`。

- 计算公式：

  $$
  y=(x1-x2)>0
  $$

## 实现原理

Greater算子通过依次调用`Ascend C`的`API`接口：`Sub`、`Mins`、`Maxs`、`Muls`，分步计算实现Greater函数计算。对于16位的数据`x1`，`x2`将其先通过`Cast`接口转换为32位浮点数后，再使用`Sub`、`Mins`、`Maxs`、`Muls`进行计算，再将结果通过`Cast`接口转换为16位浮点数。

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnGreaterGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGreater”接口执行计算。

* `aclnnStatus aclnnGreaterGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *y, uint64_t workspaceSize, aclOpExecutor **executor)`;
* `aclnnStatus aclnnGreater(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。

### aclnnGreaterGetWorkspaceSize

- **参数说明：**

  - x1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x1，数据类型支持FLOAT16、FLOAT32,BFLOAT16,INT8,UINT8,INT32,INT64，数据格式支持ND。
  - x2（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x2，数据类型支持FLOAT16、FLOAT32,BFLOAT16,INT8,UINT8,INT32,INT64，数据格式支持ND。
  - y（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出y，数据类型支持BOOL，数据格式支持ND，输出维度与x一致。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。



- **返回值：**
  
  返回aclnnStatus状态码。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x1，x2的数据类型和数据格式不在支持的范围内。
    ```

### aclnnLerp

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnFFNGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码。

## 约束与限制

- x1，x2，y，x1和x2的数据类型只支持float32,float16,bfloat16,int8,uint8,int32,int64，数据格式只支持ND，y的数据类型只支持bool，数据格式只支持ND

## 算子原型

<table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Greater</th></tr> 
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
    <tr><td rowspan="4" align="center">算子输入</td>
    <tr><td align="center">x1</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16,int8,uint8,int32,int64</td><td align="center">ND</td><tr>  
    <tr><td align="center">x2</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16,int8,uint8,int32,int64</td><td align="center">ND</td><tr></tr> 
    <tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">bool</td><td align="center">ND</td></tr>  
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">greater</td></tr>  
</table>

## 调用示例

详见[Lerp自定义算子样例说明算子调用章节](../README.md#算子调用)