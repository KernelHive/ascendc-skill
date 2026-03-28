声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# CoalesceSparse

## 支持的产品型号

Atlas A2训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

- 算子功能：将相同坐标点（indices）的value进行累加求和。
- 计算公式：
  
  $$
  newIndices[uniqueIndices[i], :] = indices[i, :], for 0 \le i \lt indices.shape[0] \\
  newValues[indices[:, v], :] = newValues[indices[:, v], :] + values[v, :], for 0 \le v \lt values.shape[0]
  $$
  
  **说明：**
  - 假设输入indices的shape为$[m, n]$，values的shape为$[n, a0, ...]$，则记$denseDim = values.dim() - 1$，按$dim = 1$对indices做max操作后再加1，得到maxIndices，将values的shape从$m$至$m+ values.dim() - 1$的元素追加到maxIndices后，得到sparseTensorShape，根据sparseTensorShape对indices做flatten，得到indicesFlatten后对其做unique操作，依次获得其余输入uniqueLen，uniqueIndices，则输出newIndices的shape为$[m, uniqueLen.shape[0]]$，newValues的shape为$[uniqueLen.shape[0], a0, ...]$。经过上述处理后将indices做$transpose(0, 1)$，才能传入进行计算，计算得到的newIndices做$transpose(0,1)$后才可与标杆一致。
  - 示例：若indices为$[[1, 1]]$，values为$[3, 4]$，则newIndices为$[[1]]$，newValues为$[7]$。

## 实现原理

CoalesceSparse由DataCopyPad、SetAtomicAdd与SetAtomicNone操作组成。

## 函数原型

每个算子分为[两段式接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用“aclnnCoalesceSparseGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCoalesceSparse”接口执行计算。

* `aclnnCoalesceSparseGetWorkspaceSize(const aclTensor* uniqueLen, const aclTensor* uniqueIndices, const aclTensor* indices, const aclTensor* values, aclTensor* newIndices, aclTensor* newValues, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnCoalesceSparse(void* workspace, int64_t workspaceSize, aclOpExecutor** executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

## aclnnCoalesceSparseGetWorkspaceSize

- **参数说明：**
  - uniqueLen（aclTensor\*，计算输入）：表示indices与values组成的sparseTensor经过flatten后的结果在unique后的唯一元素，公式中的uniqueLen，Device侧的aclTensor，数据类型支持INT64、INT32，维度支持1维，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - uniqueIndices（aclTensor\*，计算输入）：表示indices与values组成的sparseTensor经过flatten后的结果在uniqueLen中的索引，公式中的uniqueIndices，Device侧的aclTensor，数据类型支持INT64、INT32，维度支持1维，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - indices（aclTensor\*，计算输入）：表示原始输入indices，公式中的indices，Device侧的aclTensor，数据类型支持INT64、INT32，维度支持2维且indices.shape[1]必须小于64，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - values（aclTensor\*，计算输入）：表示原始输入values，公式中的values，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、INT32，维度支持1-8维，values.shape[0]需要与indices.shape[0]一致，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - newIndices（aclTensor\*，计算输出）：表示经过公式中的newIndices，Device侧的aclTensor，数据类型支持INT64、INT32，数据类型需要与indices一致，newIndices.shape[1]需要与indices.shape[1]一致，newIndices.shape[0]需要与uniqueLen.shape[0]一致，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - newValues（aclTensor\*，计算输出）：公式中的newValues，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、INT32，数据类型需要与values一致，newValues.shape[0]需要与uniqueLen.shape[0]一致，其余维度需要与values的对应维度一致，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
    ```
    第一段接口完成入参校验，出现如下场景时报错：
    返回161001（ACLNN_ERR_PARAM_NULLPTR）：uniqueLen、uniqueIndices、indices、values、new_indices或new_values是空指针。
    返回161002（ACLNN_ERR_PARAM_INVALID）：uniqueLen、uniqueIndices、indices、values、new_indices、new_values的数据类型和数据格式不在支持的范围内。
    ```

## aclnnCoalesceSparse

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnCoalesceSparseGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

假设indices的shape为[6,8], values的shape为[6, 8, 8], indices按dim=0做max的结果为[6, 6, 6, 5, 6, 6, 5, 6], 将该结果的倒序得到[6, 5, 6, 6, 5, 6, 6, 6], 然后将其视为shape，计算stride=[194400, 38880, 6480, 1080, 216, 36, 6, 1], 则对所有的$0 \le n \lt 6$都需要满足$\sum_{m=0}^{m=7} indices[n][m] * stride[m] \lt INT64\_MAX$。

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">CoalesceSparse</td></tr>
</tr>
<tr><td rowspan="5" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">unique_len</td><td align="center">1D</td><td align="center">int64, int32</td><td align="center">ND</td></tr>
<tr><td align="center">unique_indices</td><td align="center">1D</td><td align="center">int64, int32</td><td align="center">ND</td></tr>
<tr><td align="center">indices</td><td align="center">2D, [n, m]</td><td align="center">int64, int32</td><td align="center">ND</td></tr>
<tr><td align="center">values</td><td align="center">1D-8D, [n, a0, ...]</td><td align="center">float, float16, int32</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="2" align="center">算子输出</td><td align="center">new_indices</td><td align="center">2D, [unique_len.shape[0], m]</td><td align="center">int64, int32</td><td align="center">ND</td></tr>
<tr><td align="center">new_values</td><td align="center">1D-8D, [unique_len.shape[0], a0, ...]</td><td align="center">float, float16, int32</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">coalesce_sparse</td></tr>
</table>

## 调用示例

详见[CoalesceSparse自定义算子样例说明算子调用章节](../README.md#算子调用)
