声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# Less

## 支持的产品型号

- Atlas A2训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：Less算子提供比较相等的功能。Less算子的主要功能是对输入的两个数值（或张量）进行逐元素比较，判断第一个数值是否小于第二个数值。在数学和工程领域中，比较是一个基础且常见的操作，被广泛应用于图像处理、信号处理、逻辑运算等多个领域。Less算子能够高效地处理批量数据的比较，支持浮点数和整数类型的输入。

- 计算公式：

  - `half` 类型

  $$
  y\_compute =
  \begin{cases} MIN\_ACCURACY\_FP16 & \text{if }  (x2−x1)>{ MIN\_ACCURACY\_FP16}  \\
   (x2−x1)& \text{if }  (x2−x1) <{ MIN\_ACCURACY\_FP16} 
  \end{cases}
  $$

  $$
  y\_compute =
  \begin{cases}y\_compute & \text{if }  y\_compute>{ 0}  \\
   0& \text{if }  y\_compute <{ 0} 
  \end{cases}
  $$

  $$
  y=y\_compute\times MAX\_MUL\_FP16
  $$

  ​

  - 对于 `float` 类型

  $$
  y\_compute =
  \begin{cases} MIN\_ACCURACY\_FP32 & \text{if }  (x2−x1)>{ MIN\_ACCURACY\_FP32}  \\
   (x2−x1)& \text{if }  (x2−x1) <{ MIN\_ACCURACY\_FP32} 
  \end{cases}
  $$

  $$
  y\_compute =
  \begin{cases}y\_compute & \text{if }  y\_compute>{ 0}  \\
   0& \text{if }  y\_compute <{ 0} 
  \end{cases}
  $$

  $$
  y=y\_compute\times MAX\_MUL\_FP32
  $$



## 实现原理

调用`Ascend C`的`API`接口对输入数据进行计算，该实现通过计算两个输入张量的差值，接着将大于阈值的差值设置为阈值，将小于零的差值设置为 0，并乘以相关乘数。最后，根据输入数据类型将结果转换为目标类型（ `int8_t`），并存储到输出张量中。此过程支持多种数据类型，包括 `half`、`float`、`int8` 等。

## 算子执行接口

每个算子分为两段式接口，必须先调用 “aclnnLessGetWorkspaceSize” 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 “aclnnLess” 接口执行计算。

- `aclnnStatus aclnnLessGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnLess(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnLessGetWorkspaceSize

- **参数说明：**

  - x1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x1，数据类型支持FLOAT16、BFLOAT16、FLOAT32、UINT8、INT8、INT32、INT64，数据格式支持ND。
  - x2（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x2，数据类型支持FLOAT16、BFLOAT16、FLOAT32、UINT8、INT8、INT32、INT64，数据格式支持ND。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出y，数据类型支持BOOL，数据格式支持ND，输出维度与x一致。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x1、x2、out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnLess

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnFFNGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

x1，x2，out的数据类型 Atlas A2训练系列产品/Atlas 800I A2推理产品支持FLOAT16，FLOAT32，BFLOAT16，UINT8，INT8，INT32，INT64，数据格式只支持ND

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Less</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">x1</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16,uint8,int8,int32,int64</td><td align="center">ND</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">x2</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16,uint8,int8,int32,int64</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">bool</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">less</td></tr>  
</table>

## 调用示例

详见[Less自定义算子样例说明算子调用章节](../README.md#算子调用)