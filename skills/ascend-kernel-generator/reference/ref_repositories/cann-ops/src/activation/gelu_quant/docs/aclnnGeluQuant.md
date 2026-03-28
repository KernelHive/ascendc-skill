声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# aclnnGeluQuant

## 支持的产品型号

- Atlas 训练系列产品
- Atlas 推理系列产品
- Atlas A2 训练系列产品
- Atlas 800I A2 推理产品
- Atlas 200I/500 A2 推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 接口原型

每个算子分为两段式接口，必须先调用“aclnnGeluQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGeluQuant”接口执行计算。

* `aclnnStatus aclnnGeluQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *inputScaleOptional, const aclTensor *inputOffsetOptional, char *approximateOptional, char *quantModeOptional, const aclTensor *yOut, const aclTensor *outScaleOut, uint64_t *workspaceSize,
aclOpExecutor **executor)`
* `aclnnStatus aclnnGeluQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,aclrtStream stream)`

## 功能描述

GeluQuant由Gelu算子和Quant量化操作组成，计算过程：
$$
gelu = Gelu(x, approximate)
$$
$$
quantMode 是static, yOut = Round(gelu*scale+offset).clip(-128, 127)
$$
$$
quantMode 是dynamic, yOut = (gelu * scale) * (127.0 / max(abs(gelu*scale)));
$$
$$
outScale = max(abs(gelu*scale))/127.0
$$

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnGeluQuantGetWorkspaceSize

- **参数说明：**
  
  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x，数据类型支持FLOAT16，FLOAT32，BF16 数据格式支持ND，输入shape至少2维，至多8维。
  - inputScaleOptional（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，公式中的输入scale，数据类型支持FLOAT16，FLOAT32，BF16，数据格式支持ND。当quantMode是static, 则是必选参数。shape只可以是一维，大小可以是x的shape的最后一个维度，或者1。
  - inputOffsetOptional（aclTensor\*，计算输入）：可选参数，Device侧的aclTensor，公式中的输入offset，数据类型支持FLOAT16，FLOAT32，BF16，数据格式支持ND。shape和数据类型应该和inputScaleOptional一致。
  - approximateOptional（char\*，计算输入）：可选参数，公式中的输入approximate，数据类型支持STRING，数据格式。值必须是tanh或者none。
  - quantModeOptional（char\*，计算输入）：可选参数，公式中的输入quantMode，数据类型支持STRING，数据格式。值必须是dynamic或者static。
  - yOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出yOut，数据类型支持INT8，数据格式支持ND。Shape和输入x一致。
  - outScaleOut（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出outScale，数据类型支持FLOAT32，数据格式支持ND。Shape和输入x的shape除了最后一个维度，其他维度都一致。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**
  
  返回aclnnStatus状态码。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x、inputScaleOptional、inputOffsetOptional的数据类型和数据格式不在支持的范围内。
  ```

### aclnnGeluQuant

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAddCustomGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**
  
  返回aclnnStatus状态码。

## 约束与限制

x、inputScaleOptional、inputOffsetOptional的数据类型只支持FLOAT16，FLOAT32, BF16，数据格式只支持ND。