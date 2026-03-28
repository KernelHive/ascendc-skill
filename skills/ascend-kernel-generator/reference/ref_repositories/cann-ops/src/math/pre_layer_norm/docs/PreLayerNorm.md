声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# PreLayerNorm

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：`PreLayerNorm`是`Add`和`LayerNorm`的融合算子，`Add`算子的输出作为`LayerNorm`算子的输入，进行归一化处理

## 实现原理

给定两个输入张量`x`和`y`，对输入`x`, `y`先相加得到的数据，根据系数`beta`和偏置`gamma`使`Add(x, y)`的值收敛到固定区间。

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnPreLayerNormGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnPreLayerNorm”接口执行计算。

* `aclnnStatus aclnnPreLayerNormGetWorkspaceSize(const aclTensor *x, const aclTensor *y, const aclTensor *gamma, const aclTensor *beta,
double epsilon, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnPreLayerNorm(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnPreLayerNormGetWorkspaceSize

- **参数说明：**
  
  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入x，数据类型支持FLOAT，数据格式支持ND。
  - y（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入y，数据类型支持FLOAT，数据格式支持ND。
  - gamma（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入gamma，数据类型支持FLOAT，数据格式支持ND。
  - beta（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入beta，数据类型支持FLOAT，数据格式支持ND。
  - epsilon（double\*，计算输入）：可选参数，公式中的输入epsilon，数据类型支持DOUBLE，默认值1e-5。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出z，数据类型支持FLOAT16，FLOAT，INT32，数据格式支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x, y, gamma, beta, epsilon, out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnPreLayerNorm

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnPreLayerNormGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- x，y, gamma，beta，out的数据类型支持FLOAT，数据格式支持ND

## 算子原型
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">PreLayerNorm</td></tr>
</tr>
<tr><td rowspan="5" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">默认值</td></tr>
<tr><td align="center">x</td><td align="center">4980 * 4 * 2048</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">y</td><td align="center">4980 * 4 * 2048</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">gamma</td><td align="center">2048</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">beta</td><td align="center">2048</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">4980 * 4 * 2048</td><td align="center">float</td><td align="center">ND</td><td align="center">\</td></tr>
</tr>
<tr><td rowspan="1" align="center">attr属性</td><td align="center">epsilon</td><td align="center">\</td><td align="center">double</td><td align="center">\</td><td align="center">1e-5</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">pre_layer_norm</td></tr>
</table>

## 调用示例

详见[PreLayerNorm自定义算子样例说明算子调用章节](../README.md#算子调用)
