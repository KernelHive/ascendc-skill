声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# AngleV2

## 支持的产品型号

Atlas 训练系列产品/Atlas A2训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

- 算子功能：计算给定输入张量的幅角（以弧度为单位）。
- 计算公式：
  - 当输入x为复数时，其中x.real和x.imag分别代表x的实部和虚部
  $$y = \left\{
    \begin{array}{rcl}
    &atan(\frac{x.imag}{x.real}) & x.real \gt 0 \\
    &\frac{\pi}{2} & x.real = 0 \&\& x.imag \ge 0 \\
    &0 & x.real = 0 \&\& x.imag = 0 \\
    &-\frac{\pi}{2} & x.real = 0 \&\& x.imag \lt 0 \\
    &\pi + atan(\frac{x.imag}{x.real}) & x.real \lt 0 \&\& x.imag \ge 0 \\
    &-\pi + atan(\frac{x.imag}{x.real}) & x.real \lt 0 \&\& x.imag \lt 0\\
    &sign(x.imag) \times \pi & x.real = -\inf \&\& 0 \lt abs(x.imag) \lt \inf\\
    &0 & x.real = \inf \&\& 0 \lt abs(x.imag) \lt \inf\\
    &sign(x.imag) \times \frac{\pi}{4} & x.real = \inf \&\& abs(x.imag) = \inf\\
    &sign(x.imag) \times \frac{3\pi}{4} & x.real = -\inf \&\& abs(x.imag) = \inf\\
    &sign(x.imag) \times \frac{\pi}{2} & abs(x.real) \lt \inf \&\& abs(x.imag) = \inf\\
    \end{array}
    \right.
  $$

  - 当输入x为实数时
  $$y = \left\{
    \begin{array}{rcl}
    &0 & x \ge 0 \\
    &\pi & x \lt 0 \\
    \end{array}
    \right.
  $$
  **说明：**
  无。

## 实现原理

- AngleV2算子输入为complex64时，通过依次调用`Ascend C`的`API`接口：`Mul`、`Div`、`Add`、`Sub`、`Compare`、`Select`、`Duplicate`，按照公式计算。
- AngleV2算子输入为float32, float16时，通过依次调用`Ascend C`的`API`接口：`Compare`、`Select`，按照计算公式计算。
- AngleV2算子输入为bool, uint8时，通过依次调用`Ascend C`的`API`接口：`Duplicate`，填充一个全0的tensor后输出。
- AngleV2算子输入为int8, int16, int32, int64时，通过依次调用`Ascend C`的`API`接口：`Cast`、`Compare`、`Select`，按照计算公式计算，`Cast`接口将整数类型转换为32位浮点数或者16位浮点数进行计算。

## 函数原型

每个算子分为[两段式接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用“aclnnAngleV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAngleV2”接口执行计算。

* `aclnnStatus aclnnAngleV2GetWorkspaceSize(const aclTensor* x, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnAngleV2(void* workspace, uint64_t workspaceSize, aclOpExecutor** executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

## aclnnAngleV2GetWorkspaceSize

- **参数说明：**

  - x（aclTensor\*，计算输入）：表示输入数据，公式中的输入x，Device侧的aclTensor，数据类型支持FLOAT16、FLOAT、COMPLEX64、BOOL、UINT8、INT8、INT16、INT32、INT64，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - y（aclTensor\*，计算输出）：表示输出数据，公式中的输出y，Device侧的aclTensor，数据类型支持FLOAT16、FLOAT，当x的数据类型为FLOAT16时，y的数据类型为FLOAT16，否则，y的数据类型为FLOAT，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
    ```
    第一段接口完成入参校验，出现如下场景时报错：
    返回161001（ACLNN_ERR_PARAM_NULLPTR）：x或y为空指针。
    返回161002（ACLNN_ERR_PARAM_INVALID）：x、y的数据类型和数据格式不在支持的范围内。
    ```

## aclnnAngleV2

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAngleV2GetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束说明

无。

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">AngleV2</th></tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">默认值</td></tr>

<tr><td align="center">x</td><td align="center">-</td><td align="center">float32, float16, complex64, bool, uint8, int8, int16, int32, int64</td><td align="center">ND</td><td align="center">\</td></tr>

<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">-</td><td align="center">float32, float16</td><td align="center">ND</td><td align="center">\</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">angle_v2</td></td></tr>
</table>

## 调用示例

详见[AngleV2自定义算子样例说明算子调用章节](../README.md#算子调用)
