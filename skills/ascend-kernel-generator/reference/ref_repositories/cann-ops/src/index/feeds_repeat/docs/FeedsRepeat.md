声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# FeedsRepeat

## 支持的产品型号

Atlas 训练系列产品/Atlas 推理系列产品/Atlas A2训练系列产品/Atlas 800I A2推理产品/Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：对于输入feeds，根据输入feeds_repeat_times，将对应的feeds的第0维上的数据复制对应的次数，并将输出y的第0维补零到output_feeds_size的数值。

  例如对于feeds={(a, b), (c ,d), (e, f)}，feeds_repeat_times = {x, y, z}，则对应在输出里将(a, b)复制x次，(c, d)复制y次, (e, f)复制z次，若output_feeds_size = w + x + y + z，则在最后再补充w个对齐的(0, 0)；假设feeds_repeat_times = {0, 1, 2}, output_feeds_size = 4, 则对应out = {(c, d), (e, f), (e, f), (0, 0)}。

## 实现原理

1. 根据输入，计算feeds第0维每组数据在输出out中开始复制的起始位置，包括补0的起始位置，调用`Ascend C`的`API`接口`cast`、`sum`进行实现；
1. 将Feeds的第0维每组数据调用`Ascend C`的`API`接口`DataCopyPad`重复搬运到输出out中。

## 算子执行接口

* 每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnFeedsRepeatGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFeedsRepeat”接口执行计算。

  * `aclnnStatus aclnnFeedsRepeatGetWorkspaceSize(const aclTensor *feeds, const aclTensor *feedsRepeatTimes, int64_t outputFeedsSize, const aclTensor *out, uint64_t workspaceSize, aclOpExecutor **executor);`
  * `aclnnStatus aclnnFeedsRepeat(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream);`

  **说明**：

  - 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
  - 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

  ### aclnnFeedsRepeatGetWorkspaceSize

  - **参数说明：**

    - feeds（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入feeds，数据类型支持FLOAT、FLOAT16、BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
    - feedsRepeatTimes（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入feeds_repeat_times，数据类型支持INT32、INT64，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
    - outputFeedsSize (int64_t，入参)：必选参数，公式中的output_feeds_size。
    - out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出y，数据类型支持FLOAT、FLOAT16、BFLOAT16，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
    - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
    - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

  - **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

    ```
    第一段接口完成入参校验，若出现以下错误码，则对应原因为：
    - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
    - 返回161002（ACLNN_ERR_PARAM_INVALID）：x、y、out的数据类型和数据格式不在支持的范围内。
    ```

  ### aclnnFeedsRepeat

  - **参数说明：**

    - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
    - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAddCustomGetWorkspaceSize获取。
    - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

  - **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- feeds，out的数据类型只支持FLOAT32，FLOAT16，BFLOAT16，数据格式只支持ND，两者的数据类型一致；
- feeds_repeat_times的数据类型只支持INT32，INT64，数据格式只支持ND，一维tensor，长度（元素个数）必须和feeds的第0维数值相等；
- output_feeds_size为必选属性，只支持int类型的scalar，数值需大于等于feeds_repeat_times的元素总和；
- 不支持空tensor，feeds_repeat_times的数据规模（Byte大小）不能超过48KB。

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">FeedsRepeat</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">type</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">feeds</td><td align="center">tensor</td><td align="center">float32, float16, bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">feeds_repeat_times</td><td align="center">tensor</td><td align="center">int32, int64</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子属性</td><td align="center">output_feeds_size</td><td align="center">scalar</td><td align="center">int</td><td align="center">-</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">tensor</td><td align="center">float32, float16, bfloat16</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">feeds_repeat</td></tr>
</table>


## 调用示例

详见[FeedsRepeat自定义算子样例说明算子调用章节](../README.md#算子调用)
