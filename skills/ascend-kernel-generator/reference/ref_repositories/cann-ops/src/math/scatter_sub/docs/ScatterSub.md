声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# ScatterSub

## 支持的产品型号

本样例支持如下产品型号：
- Atlas A2 训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：将`var`中的数据用`indices`进行索引，索引结果与`updates`进行减法操作。具体计算方式如下：

```
# Scalar indices
var[indices, ...] -= updates[...]

# Vector indices (for each i)
var[indices[i], ...] -= updates[i, ...]

# High rank indices (for each i, ..., j)
var[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
```

- 计算公式：通过上述索引和减法操作实现对`var`的更新。


## 实现原理

ScatterSub 算子通过依次调用`Ascend C`的`API`接口，如数据拷贝`DataCopy`、数据类型转换`Cast`、减法运算`Sub`等，分步计算实现 ScatterSub 函数计算。在计算过程中，会先判断输入`Var`中需要进行减法操作的维度是否是 32B 对齐，如果是 32B 对齐，则对这些维度进行矢量减法操作；如果是 32B 非对齐，则需要将每一个标量值取出来进行标量减法操作。

## 算子执行接口

每个算子分为[两段式接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用“aclnnScatterSubGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnScatterSub”接口执行计算。

* `aclnnStatus aclnnScatterSubGetWorkspaceSize(const aclTensor* var, const aclTensor* indices, const aclTensor* updates, bool use_locking, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnScatterSub(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnScatterSubGetWorkspaceSize

- **参数说明：**
  
  - var（aclTensor*，计算输入）：必选参数，Device 侧的 aclTensor，公式中的输入`var`，数据类型支持 FLOAT16、FLOAT32、INT32、INT8，数据格式支持 ND。
  - indices（aclTensor*，计算输入）：必选参数，Device 侧的 aclTensor，用于索引`var`的张量，数据类型支持INT32，数据格式支持 ND。
  - updates（aclTensor*，计算输入）：必选参数，Device 侧的 aclTensor，用于与索引后的`var`做减法的张量，数据类型支持 FLOAT16、FLOAT32、INT32、INT8，数据格式支持 ND。
  - use_locking（bool*，算子属性）：可选参数，Host 侧的 bool，是否使用锁，默认值为 false。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：x、y、out的数据类型和数据格式不在支持的范围内。
  ```

### aclnnScatterSub

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnScatterSubGetWorkspaceSize 获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- `var`、`updates`的数据类型支持 FLOAT16、FLOAT32、INT32、INT8，数据格式只支持 ND。
- `indices`的数据类型只支持 INT32，数据格式只支持 ND。
- `use_locking`的数据类型只支持 BOOL，数据格式只支持 SCALE。

## 算子原型

<table>  
<tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">ScatterSub</th></tr>  
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>  
<tr><td align="center">var</td><td align="center">-</td><td align="center">float32,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">indices</td><td align="center">-</td><td align="center">int32</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">updates</td><td align="center">-</td><td align="center">float32,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td rowspan="1" align="center">算子输出</td><td align="center">var</td><td align="center">-</td><td align="center">float32,float16,int32,int8</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">attr属性</td><td align="center">use_locking</td><td align="center">\</td><td align="center">bool</td><td align="center">\</td><td align="center">false</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="8" align="center">scatter_sub</td></tr>  
</table>

## 调用示例

详见[ScatterSub自定义算子样例说明算子调用章节](../README.md#算子调用)
