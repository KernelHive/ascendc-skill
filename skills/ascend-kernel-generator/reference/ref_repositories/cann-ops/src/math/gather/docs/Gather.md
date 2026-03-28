声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# Gather

## 支持的产品型号

Atlas A2 训练系列产品/Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：该Gather算子提供按照输入索引数据收集数据的功能。

## 实现原理

按元素遍历索引，并调用`Ascend C`的`API`接口`DataCopy`进行数据拷贝。

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnGatherGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGather”接口执行计算。

* `aclnnStatus aclnnGatherGetWorkspaceSize(const aclTensor *x1, const aclTensor *indices, bool validateIndices, int64_t batchDims, bool isPreprocessed, bool negativeIndexSupport, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnGather(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnGatherGetWorkspaceSize

- **参数说明：**

  - x1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64，数据格式支持ND。
  - indices（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，数据类型支持INT32、INT64，数据格式支持ND。
  - validateIndices（bool，计算参数）：必选参数，传入值对算子执行无影响，数据类型为bool。
  - batchDims（int64\_t，计算参数）：必选参数，指定计算的批次维度数量，数据类型为int64\_t。
  - isPreprocessed（bool，计算参数）：必选参数，传入值对算子执行无影响，数据类型为bool。
  - negativeIndexSupport（bool，计算参数）：必选参数，传入值对算子执行无影响，数据类型为bool。
  - y（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出y，数据类型支持FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64，数据格式支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x、out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnGather

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnGatherGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。


## 约束与限制

- x，y的数据类型支持FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64，数据格式只支持ND。
- indices的数据类型支持INT32、INT64，数据格式只支持ND。

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Gather</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  

<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">x1</td><td align="center">tensor</td><td align="center">uint8,int8,uint16,int16,float16,uint32,int32,float,uint64,int64</td><td align="center">ND</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">indices</td><td align="center">tensor</td><td align="center">int32,int64</td><td align="center">ND</td></tr>  

<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">uint8,int8,uint16,int16,float16,uint32,int32,float,uint64,int64</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gather</td></tr>  
</table>

## 调用示例

详见[Gather自定义算子样例说明算子调用章节](../README.md#算子调用)