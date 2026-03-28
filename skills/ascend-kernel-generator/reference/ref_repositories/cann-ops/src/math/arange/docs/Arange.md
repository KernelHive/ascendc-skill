声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# Arange

## 支持的产品型号

- Atlas 200I/500 A2推理产品
- Atlas A2 训练系列产品
- Atlas 800I A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：从start起始到end结束按照step的间隔取值，并返回大小为 $\frac{end-start}{step}+1$的1维张量。其中，步长step是张量中相邻两个值的间隔。


- 对应的数学表达式为：
$$
\text{out}_{i+1} = \text{out}_i + \text{step}
$$

  **说明：**
  无。

## 实现原理

调用`Ascend C`的`API`接口`Duplicate`、`Mul`和`Add`进行实现。支持int32,int64,float32,float16,bfloat16五种类型，使用`TILING_KEY`分开float32类型和非float32类型的计算。

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnArangeGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnArange”接口执行计算。

* `aclnnStatus aclnnArangeGetWorkspaceSize(const aclScalar *start, const aclScalar *end, const aclScalar *step, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnArange(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnArangeGetWorkspaceSize

- **参数说明：**

  - start(aclScalar,计算输入)：Host侧的aclScalar，获取值的范围的起始位置，数据类型支持FLOAT16、FLOAT、INT32、INT64、BFLOAT16，数据格式支持ND。需要满足在step大于0时输入的start小于end，或者step小于0时输入的start大于end。
  - end(aclScalar,计算输入)：Host侧的aclScalar，获取值的范围的结束位置，数据类型支持FLOAT16、FLOAT、INT32、INT64、BFLOAT16，数据格式支持ND。需要满足在step大于0时输入的start小于end，或者step小于0时输入的start大于end。
  - step(aclScalar,计算输入)：Host侧的aclScalar，获取值的步长，数据类型支持FLOAT16、FLOAT、INT32、INT64、BFLOAT16，数据格式支持ND。需要满足step不等于0。
  - out(aclTensor，计算输出)：Device侧的aclTensor；数据类型支持FLOAT16、FLOAT、INT32、INT64、BFLOAT16，数据格式支持ND。
  - workspaceSize(uint64_t*，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：start、end、step或out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnArange

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnArangeGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- start、end、step、out的数据类型只支持FLOAT16、FLOAT、INT32、INT64、BFLOAT16，数据格式只支持ND

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Arange</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="4" align="center">算子输入</td>
<tr><td align="center">start</td><td align="center">scalar</td><td align="center">int32,int64,float32,float16,bfloat16</td><td align="center">ND</td></tr>  

<tr><td align="center">end</td><td align="center">scalar</td><td align="center">int32,int64,float32,float16,bfloat16</td><td align="center">ND</td></tr> 

<tr><td align="center">step</td><td align="center">scalar</td><td align="center">int32,int64,float32,float16,bfloat16</td><td align="center">ND</td></tr> 

<tr><td rowspan="1" align="center">算子输出</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">int32,int64,float32,float16,bfloat16</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">arange</td></tr>  
</table>

## 调用示例

详见[Arange自定义算子样例说明算子调用章节](../README.md#算子调用)
