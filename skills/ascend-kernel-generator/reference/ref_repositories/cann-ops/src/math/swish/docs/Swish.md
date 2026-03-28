声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# Swish

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。


## 功能描述

- 算子功能：该算子实现Swish激活函数计算功能。Swish激活函数定义为输入 $x$ 与 $Sigmoid(s*x)$ 函数值的乘积。该函数结合线性与非线性特性，实现更平滑的梯度传播，其平滑曲线允许部分负值输入保留，能够在深层网络中优化训练过程并提高模型性能。通过调整参数 $s$ 取值，Swish可以在线性函数和ReLU函数之间平滑过渡：当 $s\to\infty$时，Swish趋近于ReLU；当 $s=0$时，Swish成为简单的缩放线性函数。Swish 函数具有非单调特性，其梯度在不同区间表现不同，将其应用在训练深度神经网络中能够有效避免梯度消失问题，并且在优化过程中有良好的自适应性，有助于模型更好地拟合数据，在许多深度学习任务中发挥重要作用。
- 计算公式：
    $$y=x\cdot\mathrm{sigmoid}\left(s\cdot x\right)=x\cdot\frac{1}{1+e^{-s\cdot x}}$$

## 实现原理

Swish算子通过依次调用`Ascend C`的`API`接口：`Muls`、`Exp`、`Adds`、`Div`，分步计算实现Swish激活函数计算。对于16位的数据类型将其通过`Cast`接口转换为32位浮点数进行计算。

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnSwishGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwish”接口执行计算。

* `aclnnStatus aclnnSwishGetWorkspaceSize(const aclTensor* self, const aclScalar* scale, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnSwish(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnSwishGetWorkspaceSize

- **参数说明：**

  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
  - scale（aclScalar\*, 算子输入）：可选参数，Host侧的float，公式中的参数s，$Sigmoid$函数比例系数，支持数据类型为float，默认数值为1.0。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出y，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND，输出维度与x一致。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。



- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x、scale、out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnSwish

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnFFNGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。


## 约束与限制

- x，out的数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式只支持ND

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Swish</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="3" align="center">算子输入</td>
 
<tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  

<tr><td align="center">scale</td><td align="center">scalar</td><td align="center">float32,float32,float32</td><td align="center">ND</td></tr> 

<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
 
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">swish</td></tr>  
</table>

## 调用示例

详见[Swish自定义算子样例说明算子调用章节](../README.md#算子调用)