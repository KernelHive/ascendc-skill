声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# GeluGrad

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：该AscendC算子用于计算Gelu函数的梯度。

- 计算公式：

  - **Gelu函数**

    $$
    y=\frac{x}{exp((c_{0}x^{2}+c_{1})x)+1}
    $$
    其中，$c_{0}=-0.0713548162726002527220,c_{1}=-1.595769121605730711759$
  - **对于Atlas A2 训练系列产品:**

    $$
    px=exp((x^{2}\times c_{0}+c_{1})\times x)
    $$
    $$
    res0=(x^{2}\times c_{2}+c_{3})\times x
    $$
    $$
    t=\frac{1}{px+1}
    $$
    $$
    z=(px\times res0\times t^{2}+t)\times dy
    $$
    其中，$c_{0}=-0.0713548162726002527220,c_{1}=-1.595769121605730711759,c_{2}=0.2140644488178007,c_{3}=1.595769121605730711759$
  - **对于Atlas 200I/500 A2推理产品：**

    $$
    g1=\frac{1}{exp((x^{2}\times c_{0}+c_{1})\times x)+1}
    $$
    $$
    g2=x^{2}\times c_{2}+c_{3}
    $$
    $$
    z=((((g1-1)\times x)\times g2+1)\times g1)\times dy
    $$
    其中，$c_{0}=-0.0713548162726002527220,c_{1}=-1.5957691216057308,c_{2}=-0.21406444881780074632901625683959062,c_{3}=-1.5957691216057307117597842397375274738$

## 实现原理

调用`Ascend C`的`API`接口`Mul`、`Muls`、`Add`、`Adds`、`Exp`、`Duplicate`和`Div`进行实现。对于16位的数据类型将其通过`Cast`接口转换为32位浮点数进行计算。


## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnGeluGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGeluGrad”接口执行计算。

* `aclnnStatus aclnnGeluGradGetWorkspaceSize(const aclTensor* dy, const aclTensor* x, const aclTensor* y, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnGeluGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnGeluGradGetWorkspaceSize

- **参数说明：**

  - dy（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入dy，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
  - y（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入y，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出z，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND，输出维度与x一致。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：dy、x、y、out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnGeluGrad

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnGeluGradGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。


## 约束与限制

- dy，x，y，out的数据类型支持BFLOAT16，FLOAT16，FLOAT32，数据格式只支持ND

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">GeluGrad</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">dy</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td>
</tr> 
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td>
</tr> 
<tr><td rowspan="1" align="center">算子输入</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">z</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>    
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">gelu_grad</td></tr>  
</table>

## 调用示例

详见[GeluGrad自定义算子样例说明算子调用章节](../README.md#算子调用)