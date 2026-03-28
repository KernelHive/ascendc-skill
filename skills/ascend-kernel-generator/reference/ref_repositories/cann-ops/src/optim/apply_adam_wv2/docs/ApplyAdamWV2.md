声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# ApplyAdamWV2

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。


## 功能描述

- **算子功能：** 实现adamW优化器功能。

- **计算公式：**
  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
  $$
  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$
  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}} \\
  $$
  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}} \\
  $$
  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnApplyAdamWV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnApplyAdamWV2”接口执行计算。

* `aclnnStatus aclnnApplyAdamWV2GetWorkspaceSize(aclTensor *varRef, aclTensor *mRef, aclTensor *vRef, aclTensor *maxGradNormOptionalRef, const aclTensor *grad, const aclTensor *step, float lr, float beta1, float beta2, float weightDecay, float eps, bool amsgrad, bool maximize, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnApplyAdamWV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。

## aclnnApplyAdamWV2GetWorkspaceSize

- **参数说明：**

  * varRef（aclTensor\*, 计算输入/计算输出）：待计算的权重输入同时也是输出，公式中的theta，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  * mRef（aclTensor\*, 计算输入/计算输出）：adamw优化器中m参数，公式中的m，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32，shape、dtype要求与“varRef”参数一致。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  * vRef（aclTensor\*, 计算输入/计算输出）：adamw优化器中v参数，公式中的v，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32，shape、dtype要求与“varRef”参数一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  * maxGradNormOptionalRef（aclTensor\*, 计算输入/计算输出）：保存v参数的最大值，公式中的v，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32，shape要求与“varRef”参数一致，此参数在amsgrad参数为true时必选，在amsgrad参数为false时可选。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  * grad（aclTensor*, 计算输入）：梯度数据，公式中的gt，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32，shape要求与“varRef”参数一致。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  * step（aclTensor*, 计算输入）：迭代次数，公式中的t，Device侧的aclTensor，数据类型支持INT64、FLOAT32，元素个数为1，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  * lr（float\*, 计算输入）：学习率，公式中的eta，数据类型支持FLOAT。
  * beta1（float\*, 计算输入）：beta1参数，数据类型支持FLOAT。
  * beta2（float\*, 计算输入）：beta2参数，数据类型支持FLOAT。
  * weightDecay（float\*, 计算输入）：权重衰减系数，数据类型支持FLOAT。
  * eps（float\*, 计算输入）：防止除数为0，数据类型支持FLOAT。
  * amsgrad（bool, 计算输入）：是否使用算法的AMSGrad变量，数据类型为BOOL。
  * maximize（bool, 计算输入）：是否最大化参数，数据类型为BOOL。
  * workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor\*\*, 出参）：内存地址。

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的varRef、mRef、vRef、maxGradNormOptionalRef、grad、step是空指针时。
  161002 (ACLNN_ERR_PARAM_INVALID)：1. varRef、mRef、vRef、maxGradNormOptionalRef、grad、step的数据类型不在支持的范围内时。
                                    2. varRef、mRef、vRef、maxGradNormOptionalRef、grad、step的数据格式不在支持的范围内时。
                                    3. mRef、vRef、grad和varRef的shape不一致时。
                                    4. 当amsgrad为true时，maxGradNormOptionalRef和varRef的shape不一致时
                                    5. step的shape大小不为1时。
  ```

## aclnnApplyAdamWV2

- **参数说明：**

  * workspace(void \*, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnApplyAdamWV2GetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- 输入张量中varRef、mRef、vRef的数据类型一致时，数据类型支持FLOAT16、BFLOAT16、FLOAT32。

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">ApplyAdamWV2</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入/输出</td> 
<tr><td align="center">varRef</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入/输出</td>
<tr><td align="center">mRef</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入/输出</td>
<tr><td align="center">vRef</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入/输出</td>
<tr><td align="center">maxGradNormOptionalRef</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  

<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">grad</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">step</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  

<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">lr</td><td align="center">scalar</td><td align="center">float</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">beta1</td><td align="center">scalar</td><td align="center">float</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">beta2</td><td align="center">scalar</td><td align="center">float</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">weightDecay</td><td align="center">scalar</td><td align="center">float</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">eps</td><td align="center">scalar</td><td align="center">float</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">amsgrad</td><td align="center">scalar</td><td align="center">bool</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">maximize</td><td align="center">scalar</td><td align="center">bool</td><td align="center">-</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">apply_adam_w_v2</td></tr>  
</table>

## 调用示例

详见[ApplyAdamWV2自定义算子样例说明算子调用章节](../README.md#算子调用)