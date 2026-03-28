声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# MoeSoftMaxTopk

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：`MoeSoftMaxTopk`是`softmax`和`topk`的融合算子。

## 实现原理

给定两个输入张量`x`, 首先通过`softmax`对x计算最后一维每个数据的概率，对计算结果使用`topk`筛选出k个最大结果，输出对应的y值和索引indices

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnMoeSoftMaxTopkGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeSoftMaxTopk”接口执行计算。

* `aclnnStatus aclnnMoeSoftMaxTopkGetWorkspaceSize(const aclTensor *x, int64_t k, const aclTensor *yOut, const aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeSoftMaxTopk(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnMoeSoftMaxTopkGetWorkspaceSize

- **参数说明：**
  
  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x，数据类型支持FLOAT，数据格式支持ND。
  - k（int\*，计算输入）：可选参数，公式中的输入k，数据类型支持INT64，默认值4。
  - yOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出y，数据类型支持INT32，数据格式支持ND。
  - indicesOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出indices，数据类型支持INT32，数据格式支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x, y, gamma, beta, epsilon, out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnMoeSoftMaxTopk

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeSoftMaxTopkGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- x，y, indices的数据类型支持FLOAT，数据格式支持ND

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">MoeSoftMaxTopk</td></tr>
</tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>
<tr><td align="center">x</td><td align="center">1024 * 16</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>

</tr>
</tr>
<tr><td rowspan="3" align="center">算子输出</td>

<tr><td align="center">y</td><td align="center">1024 * 4</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">indices</td><td align="center">1024 * 4</td><td align="center">int</td><td align="center">ND</td><td align="center">\</td></tr>
</tr>
<tr><td rowspan="1" align="center">attr属性</td><td align="center">k</td><td align="center">\</td><td align="center">int</td><td align="center">\</td><td align="center">4</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">moe_soft_max_topk</td></tr>
</table>

## 调用示例

详见[MoeSoftMaxTopk自定义算子样例说明算子调用章节](../README.md#算子调用)
