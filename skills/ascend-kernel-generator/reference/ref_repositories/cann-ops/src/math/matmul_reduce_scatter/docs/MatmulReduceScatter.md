声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# MatmulReduceScatter

## 支持的产品型号

Atlas A2训练系列产品/Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：实现了Matmul矩阵乘法运算操作和ReduceScatter通信操作融合的功能。
- 计算公式：
  
  $$
  y=ReduceScatter(x1 * x2）
  $$
  
  **说明：**
  - ReduceScatter() 为集合通信ReduceScatter通信操作。
  - x1、x2为源操作数，x1为左矩阵，形状为\[M, K]；x2为右矩阵，形状为\[K, N]。
  - y为目的操作数，存放ReduceScatter通信结果的矩阵，形状为[M / rankDim, N]，其中rankDim为通信域内的节点数。

## 实现原理

MatmulReduceScatter由Matmul和ReduceScatter操作组成，计算过程分为2步：

1. out = Matmul(x1, x2)
2. y = ReduceScatter(out)

## 算子执行接口

每个算子分为[两段式接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用“aclnnMatmulReduceScatterGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMatmulReduceScatter”接口执行计算。

* `aclnnStatus aclnnMatmulReduceScatterGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional, char *group, char *reduceOpOptional, bool isTransA, bool isTransB, int64_t commTurn, const aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnMatmulReduceScatter(void *workspace, int64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的方式调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnMatmulReduceScatterGetWorkspaceSize

- **参数说明：**
  
  - x1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x1，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - x2（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x2，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - biasOptional （aclTensor*，偏置） 可选参数，偏置项。
  - group （char*，通信组） 必选参数，指定ReduceScatter操作的通信组名称。
  - reduceOpOptional （char*，规约操作） 可选参数，指定ReduceScatter的规约操作类型，默认为"sum"。
  - isTransA （bool，转置标志） 可选参数，指定x1是否需要转置，true表示需要转置，false表示不转置。默认为false。
  - isTransB （bool，转置标志） 可选参数，指定x2是否需要转置，true表示需要转置，false表示不转置。默认为false。
  - commTurn （int64_t，通信轮次） 可选参数，指定通信的优先级。默认为0。
  - y（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出z，数据类型支持FLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x1、x2、y的数据类型和数据格式不在支持的范围内。
  ```

### aclnnMatmulReduceScatter

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMatmulReduceScatterGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- x1，x2，y的数据类型只支持FLOAT16，数据格式只支持ND

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">MatmulReduceScatter</td></tr>
</tr>
<tr><td rowspan="9" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x1</td><td align="center">16384 * 640</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">x2</td><td align="center">640 * 5120</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">bias</td><td align="center">/</td><td align="center">/</td><td align="center">/</td></tr>
<tr><td align="center">group</td><td align="center">/</td><td align="center">char *</td><td align="center">/</td></tr>
<tr><td align="center">reduceOpOptional</td><td align="center">/</td><td align="center">char *  </td><td align="center">/</td></tr>
<tr><td align="center">isTransA</td><td align="center">/</td><td align="center">bool </td><td align="center">/</td></tr>
<tr><td align="center">isTransB</td><td align="center">/</td><td align="center">bool </td><td align="center">/</td></tr>
<tr><td align="center">commTurn</td><td align="center">/</td><td align="center">int64_t </td><td align="center">/</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">16384 * 5120</td><td align="center">float16</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_reduce_scatter</td></tr>
</table>

## 调用示例

详见[MatmulReduceScatter自定义算子样例说明算子调用章节](../README.md#算子调用)
