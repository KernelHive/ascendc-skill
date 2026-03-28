声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# ScatterReduce

## 支持的产品型号

- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- **算子功能**：将源张量（`src`）的值按照索引（`index`）规则归约到目标张量（`x`）的指定维度（`dim`），支持多种归约操作

- **归约操作支持**：`sum`（求和）、`prod`（求积）、`mean`（平均值）、`amax`（最大值）、`amin`（最小值）

- **计算公式**：
  ```python
  if include_self is True:
    y[index, ...] = reduce_op(x[index, ...], src[...])
  else:
    if x[index, ...] == None:
      y[index, ...] = src[...]
    else:
      y[index, ...] = reduce_op(x[index, ...])
  ```

## 实现原理

- **归约操作实现**：
  在y的Localtensor上进行批次的归约操作。对于每个需要规约的批次，基于`index`利用`gather`获取对应位置的原数值，调用`Sum`、`Mul`、`Max`、`Min`等接口实现向量化的规约操作，利用`SetValue`将规约后的结果放回对应的位置。对于`mean`规约操作，还需要额外统计每个位置的规约元素数量`Cnt`，在`sum`的基础上完成平均值的计算。

- **`include_self`实现**：当`include_self`为`True`时，利用`x`初始化`y`即可。当`include_self`为`False`时，利用每个位置的规约元素数量`Cnt`判断对应位置是否需要规约。

- **额外说明**：
  本算子在实现基础功能的基础上，仅实现了部分条件下的向量化加速，用户可根据示例完善其他条件下的加速。

## 算子执行接口

采用[两段式接口](common/两段式接口.md)，必须先调用"aclnnScatterReduceGetWorkspaceSize"获取工作空间，再调用"aclnnScatterReduce"执行计算。

* `aclnnStatus aclnnScatterReduceGetWorkspaceSize(const aclTensor *x, const aclTensor *index, const aclTensor *src, const int dim, const char *reduce, const bool include_self, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnScatterReduce(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

### aclnnScatterReduceGetWorkspaceSize

- **参数说明**：
  - `x` (aclTensor*, 输入)：目标张量，数据类型支持FP32/FP16，数据格式支持ND
  - `index` (aclTensor*, 输入)：索引张量，数据类型为INT32，数据格式支持ND
  - `src` (aclTensor*, 输入)：源张量，数据类型支持FP32/FP16，数据格式支持ND
  - `dim` (int, 属性)：归约操作的维度
  - `reduce` (const char*, 属性)：归约操作类型（"sum"/"mean"/"amax"/"amin"等）
  - `include_self` (bool, 属性)：是否包含原始值参与归约，默认为true
  - `workspaceSize` (uint64_t*, 输出)：返回所需工作空间大小
  - `executor` (aclOpExecutor**, 输出)：返回算子执行器

- **返回值**：
  - 返回`aclnnStatus`状态码（参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)）
  - 错误码说明：
    - `161001` (ACLNN_ERR_PARAM_NULLPTR)：必选参数为空指针
    - `161002` (ACLNN_ERR_PARAM_INVALID)：数据类型/格式不支持或归约操作无效

### aclnnScatterReduce

- **参数说明**：
  - `workspace` (void*, 输入)：工作空间内存地址
  - `workspaceSize` (uint64_t, 输入)：工作空间大小
  - `executor` (aclOpExecutor*, 输入)：算子执行器
  - `stream` (aclrtStream, 输入)：AscendCL流

- **返回值**：
  - 返回`aclnnStatus`状态码

## 约束与限制

1. **数据类型约束**：
   - `x`和`src`：支持FP32、FP16
   - `index`：仅支持INT32
2. **数据格式**：所有张量仅支持ND格式
3. **索引范围**：索引值必须在目标张量维度范围内

## 算子原型

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">ScatterReduce</th></tr>
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td rowspan="3" align="center">算子输入</td>
    <td align="center">x</td><td align="center">tensor</td><td align="center">fp32, fp16</td><td align="center">ND</td></tr>
    <td align="center">index</td><td align="center">tensor</td><td align="center">int32</td><td align="center">ND</td></tr>
    <td align="center">src</td><td align="center">tensor</td><td align="center">fp32, fp16</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">fp32, fp16</td><td align="center">ND</td></tr>
<tr><td rowspan="3" align="center">attr属性</td>
    <td align="center">dim</td><td align="center">int</td><td colspan="2" align="center">required</td></tr>
    <td align="center">reduce</td><td align="center">str</td><td colspan="2" align="center">required</td></tr>
    <td align="center">include_self</td><td align="center">bool</td><td colspan="2" align="center">default: TRUE</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">scatter_reduce</td></tr>
</table>

## 调用示例

详见[ScatterReduce自定义算子样例说明算子调用章节](../README.md#算子调用)
</details>
