声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# ApplyFusedEmaAdam

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。


## 功能描述

- **算子功能**：实现FusedEmaAdam融合优化器功能。

- **计算公式**：

  $$
  (correction_{\beta_1},correction_{\beta_2},)=\begin{cases}
  (1,1),&biasCorrection=False\\
  (1-\beta_1^{step},1-\beta_2^{step}),&biasCorrection=True
  \end{cases}
  $$
  
  $$
  grad=\begin{cases}
  grad+weightDecay*var,&mode=0\\
  grad,&mode=1
  \end{cases}
  $$
  
  $$
  m_{out}=\beta_1*m+(1-\beta_1)*grad
  $$

  $$
  v_{out}=\beta_2*v+(1-\beta_2)*grad^2
  $$

  $$
  m_{next}=m_{out}/correction_{\beta_1}
  $$

  $$
  v_{next}=v_{out}/correction_{\beta_2}
  $$

  $$
  denom=\sqrt{v_{next}}+eps
  $$

  $$
  update=\begin{cases}
  m_{next}/denom,&mode=0\\
  m_{next}/denom+weightDecay*var,&mode=1
  \end{cases}
  $$

  $$
  var_{out}=var-lr*update
  $$

  $$
  s_{out}=emaDecay*s+(1-emaDecay)*var_{out}
  $$

## 算子执行接口

每个算子分为两段式接口，必须先调用 “aclnnApplyFusedEmaAdamGetWorkspaceSize” 接口获取入参并根据计算流程计算所需workspace大小以及包含了算子计算流程的执行器，再调用 “aclnnApplyFusedEmaAdam” 接口执行计算。

* `aclnnStatus aclnnApplyFusedEmaAdamGetWorkspaceSize(const aclTensor* grad, aclTensor* varRef, aclTensor* mRef, aclTensor* vRef, aclTensor* sRef, const aclTensor* step, double lr, double emaDecay, double beta1, double beta2, double eps, int64_t mode, bool biasCorrection, double weightDecay, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnApplyFusedEmaAdam(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

## aclnnApplyFusedEmaAdamGetWorkspaceSize

- **参数说明：**

  - grad（aclTensor*，计算输入）：待更新参数对应的梯度，对应公式中的``grad``，Device侧的aclTensor，数据类型支持BFLOAT16，FLOAT16，FLOAT32，数据格式支持ND。
  - varRef（aclTensor\*，计算输入/输出）：待更新参数，对应公式中的``var``，Device侧的aclTensor, 数据类型支持BFLOAT16，FLOAT16，FLOAT32，shape和数据类型需要和grad保持一致，数据格式支持ND。
  - mRef（aclTensor\*，计算输入/输出）：待更新参数对应的一阶动量，对应公式中的``m``，Device侧的aclTensor, 数据类型支持BFLOAT16，FLOAT16，FLOAT32，shape和数据类型需要和grad保持一致，数据格式支持ND。
  - vRef（aclTensor\*，计算输入/输出）：待更新参数对应的二阶动量，对应公式中的``v``，Device侧的aclTensor, 数据类型支持BFLOAT16，FLOAT16，FLOAT32，shape和数据类型需要和grad保持一致，数据格式支持ND。
  - sRef（aclTensor\*，计算输入/输出）：待更新参数对应的EMA权重，对应公式中的``s``，Device侧的aclTensor, 数据类型支持BFLOAT16，FLOAT16，FLOAT32，shape和数据类型需要和grad保持一致，数据格式支持ND。
  - step（aclTensor*，计算输入）：优化器当前的更新次数，对应公式中的``step``，Device侧的aclTensor, 数据类型支持INT64，数据格式支持ND。
  - lr（double，计算输入）：学习率，对应公式中的``lr``。
  - emaDecay（double，计算输入）：指数移动平均（EMA）的衰减速率，对应公式中的``emaDecay``。
  - beta1（double，计算输入）：计算一阶动量的系数，对应公式中的$\beta_1$。
  - beta2（double，计算输入）：计算二阶动量的系数，对应公式中的$\beta_2$。
  - eps（double，计算输入）：加到分母上的项，用于数值稳定性，对应公式中的``eps``。
  - mode（int64_t，计算输入）：控制应用L2正则化还是权重衰减，对应公式中的``mode``，1为adamw，0为L2。
  - biasCorrection（bool，计算输入）：控制是否进行偏差校正，对应公式中的``biasCorrection``，true表示进行校正，false表示不做校正。
  - weightDecay（double，计算输入）：权重衰减，对应公式中的``weightDecay``。
  - workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. 输入和输出的Tensor是空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. 输入和输出的数据类型和数据格式不在支持的范围之内；
                                       2. 输入grad、var、m、v、s的数据类型和shape不一致。
  ```

## aclnnApplyFusedEmaAdam

- **参数说明：**
  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnApplyFusedEmaAdamGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- 输入张量中varRef、mRef、vRef的数据类型一致时，数据类型支持FLOAT16、BFLOAT16、FLOAT32。

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">ApplyAdamWV2</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入/输出</td> 
<tr><td align="center">grad</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入/输出</td> 
<tr><td align="center">varRef</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入/输出</td>
<tr><td align="center">mRef</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入/输出</td>
<tr><td align="center">vRef</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入/输出</td>
<tr><td align="center">sRef</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">step</td><td align="center">tensor</td><td align="center">int64</td><td align="center">ND</td></tr>  

<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">lr</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">emaDecay</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">beta1</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">beta2</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">eps</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">mode</td><td align="center">scalar</td><td align="center">int64</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">biasCorrection</td><td align="center">scalar</td><td align="center">bool</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">weightDecay</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">apply_fused_ema_adam</td></tr>  
</table>

## 调用示例

详见[ApplyFusedEmaAdam自定义算子样例说明算子调用章节](../README.md#算子调用)