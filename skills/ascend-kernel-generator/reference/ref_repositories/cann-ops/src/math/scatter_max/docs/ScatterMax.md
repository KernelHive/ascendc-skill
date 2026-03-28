声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# ScatterMax

## 支持的产品型号

- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：将输入张量的值按照指定维度进行聚合，并返回指定维度上的最大值和最大值所在的索引。
- 计算公式：
  ```python
  # 标量索引
  var[indices, ...] = max(var[indices, ...], updates[...])

  # 向量索引（对于每个 i）
  ref[indices[i], ...] = max(var[indices[i], ...], updates[i, ...])

  # 高维索引（对于每个 i, ..., j）
  var[indices[i, ..., j], ...] = max(var[indices[i, ..., j], ...],
  updates[i, ..., j, ...])
  ```

## 实现原理

- 使用原子操作确保并发更新正确性： AscendC::SetAtomicMax<T_VAR>()
- 数据管理与传输： AscendC::LocalTensor / GlobalTensor 、 AscendC::DataCopy
- 内存管理： AscendC::TPipe 、 AscendC::Enque和AscendC::Deque 等队列操作API

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnScatterMaxGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnScatterMax”接口执行计算。

* `aclnnStatus aclnnScatterMaxGetWorkspaceSize(const aclTensor *var, const aclTensor *indice, const alcTensor *updates, bool use_locking, uint64_t workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnScatterMax(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnScatterMaxGetWorkspaceSize

- **参数说明：**
  
  - var（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入var，数据类型支持FLOAT32\FLOAT16\INT32\INT8，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - indice（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入indice，数据类型支持INT32，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - updates（aclTensor\*，计算输入）：Device侧的aclTensor，公式中的输入updates，数据类型支持FLOAT32\FLOAT16\INT32\INT8，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - use_locking（bool，计算输入）：Device侧的属性，公式中的输入use_locking，数据类型支持bool。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：var、indice、update、use_locking的数据类型和数据格式不在支持的范围内。
  ```

### aclnnScatterMax

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAddGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- var，indice，updates的数据类型只支持FLOAT32\FLOAT16\INT32\INT8，数据格式只支持ND

## 算子原型

<table>  
<tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">ScatterMax</th></tr>  
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>
<tr><td align="center">var</td><td align="center">-</td><td align="center">float32,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">indices</td><td align="center">-</td><td align="center">int32,int32,int32,int32</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">updates</td><td align="center">-</td><td align="center">float32,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td rowspan="1" align="center">attr属性</td><td align="center">use_locking</td><td align="center">\</td><td align="center">bool</td><td align="center">\</td><td align="center">false</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">scatter_max</td></tr>  
</table>

## 调用示例

详见[ScatterMax自定义算子样例说明算子调用章节](../README.md#算子调用)
