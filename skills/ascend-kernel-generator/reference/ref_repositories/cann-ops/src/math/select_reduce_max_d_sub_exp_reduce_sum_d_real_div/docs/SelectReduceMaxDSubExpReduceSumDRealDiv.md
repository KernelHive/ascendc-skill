声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# SelectReduceMaxDSubExpReduceSumDRealDiv

## 支持的产品型号

Atlas A2训练系列产品/Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：实现了数据select + reduce + max + sub + exp + reduce_sum_d + div，返回结果的功能。
- 计算公式：
  
  $$\text{input1_sel} = \text{input1} \cdot \text{sel} $$
  $$\text{input2_sel} = \text{input2} \cdot (\neg \text{sel}) $$
  $$\text{add_res} = \text{input1_sel} + \text{input2_sel} $$
  $$\text{max_res} = \max(\text{add_res}, \text{axis}=-1) $$
  $$\text{sub_res} = \text{add_res} - \text{max_res} $$
  $$\text{exp_res} = e^{\text{sub_res}} $$
  $$\text{sum_res} = \sum(\text{exp_res}, \text{axis}=-1) $$
  $$\text{result} = \frac{\text{exp_res}}{\text{sum_res}} $$
  $$\text{output}_1 = \frac{1}{1 + e^{-(\text{sub_res} \cdot 1)}} $$
  
  **说明：**
  无。

## 实现原理

SelectReduceMaxDSubExpReduceSumDRealDiv由select + reduce + max + sub + exp + reduce_sum_d + div操作组成，计算过程：

1. input1_sel = input1 * sel
2. input2_sel = input2 * (~sel)
3. reduce_res = input1_sel + input2_sel
4. max_res = np.max(mul_res, axis=-1, keepdims=True)
5. sub_res = mul_res - max_res
6. exp_res = np.exp(sub_res)
7. sum_res = np.sum(exp_res, axis=-1, keepdims=True)
8. output = exp_res / sum_res

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnSelectReduceMaxDSubExpReduceSumDRealDivGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSelectReduceMaxDSubExpReduceSumDRealDiv”接口执行计算。

* `aclnnStatus aclnnSelectReduceMaxDSubExpReduceSumDRealDivGetWorkspaceSize(const aclTensor *sel, const aclTensor *input1, const aclTensor *input2, const aclTensor *output, uint64_t workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnSelectReduceMaxDSubExpReduceSumDRealDiv(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnSelectReduceMaxDSubExpReduceSumDRealDivGetWorkspaceSize

- **参数说明：**
  
  - sel（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入sel，数据类型支持BOOL，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - input1（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入input1，数据类型支持FLOAT16/FLOAT32/BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - input2（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入input2，数据类型支持FLOAT16/FLOAT32/BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - output（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出output，数据类型支持FLOAT16/FLOAT32/BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

### aclnnSelectReduceMaxDSubExpReduceSumDRealDiv

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnSelectReduceMaxDSubExpReduceSumDRealDivGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- input1，input2，output的数据类型只支持FLOAT16/FLOAT32/BFLOAT16, sel的数据类型只支持BOOL，数据格式只支持ND

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">SelectReduceMaxDSubExpReduceSumDRealDiv</td></tr>
</tr>
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">sel</td><td align="center">64 * 1 * 60</td><td align="center">bool</td><td align="center">ND</td></tr>
<tr><td align="center">input1</td><td align="center">64 * 1 * 60</td><td align="center">float16/float32/bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">input2</td><td align="center">64 * 1 * 60</td><td align="center">float16/float32/bfloat16</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">output</td><td align="center">64 * 1 * 60</td><td align="center">float16/float32/bfloat16</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">select_reduce_max_d_sub_exp_reduce_sum_d_real_div</td></tr>
</table>

## 调用示例

详见[SelectReduceMaxDSubExpReduceSumDRealDiv自定义算子样例说明算子调用章节](../README.md#算子调用)
