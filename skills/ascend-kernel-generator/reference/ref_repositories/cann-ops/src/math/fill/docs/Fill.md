声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# Fill

## 支持的产品型号

Atlas A2训练系列产品 / Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

算子功能

-  Fill算子创建一个形状由输入dims指定的张量，并用标量值value填充所有元素。该算子常用于初始化张量为特定值。

## 实现原理

调用`Ascend C`的`API`接口`Duplicate`进行实现。

- 对于INT8和BOOL，将value其通过`Cast`接口转换为FLOAT16后进行填充，再通过`Cast`接口转化为原来的数据类型。
- 对于INT32、FLOAT16、FLOAT32、BFLOAT16，直接进行填充。
- 对于INT64，将value视为两个INT32组证，分别填充前32位和后32位。

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnFillGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFill”接口执行计算。

* `aclnnStatus aclnnFillGetWorkspaceSize(const aclIntArray *dims, const aclScalar *value, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnFill(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnFillGetWorkspaceSize

- **参数说明：**

  - aclIntArray\*，计算输入）：必选参数，Device侧的aclIntArray，公式中的输入x，数据类型支持FLOAT16、BFLOAT16、FLOAT32。
  - value（aclTensor\*，计算输出）：Device侧的aclScalar，公式中的输出y，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT8、INT32、INT64、BOOL。
  - y （aclTensor\*, 计算输出）：Device侧的aclTensor，公式中的输出y，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT8、INT32、INT64、BOOL，数据格式支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x、out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnFill

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnFillGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。


## 约束与限制

- dims的数据类型支持INT64。
- value的数据类型支持FLOAT16、FLOAT32、INT8、INT32、INT64，BOOL、BFLOAT16。
- y的数据类型支持FLOAT16、FLOAT32、INT8、INT32、INT64，BOOL、BFLOAT16。，数据格式只支持ND。

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Fill</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td>
 
<tr><td align="center">dims</td><td align="center">attr_tuple</td><td align="center">int64</td><td align="center">-</td></tr>  

<tr><td rowspan="2" align="center">算子输入</td>
 
<tr><td align="center">value</td><td align="center">scalar</td><td align="center">float32,float16,bfloat16,int8,bool,int64,int32</td><td align="center">-</td></tr>  

<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16,int8,bool,int64,int32</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">Fill</td></tr>  
</table>

## 调用示例

详见[Fill自定义算子样例说明运行验证章节](../README.md#运行验证)