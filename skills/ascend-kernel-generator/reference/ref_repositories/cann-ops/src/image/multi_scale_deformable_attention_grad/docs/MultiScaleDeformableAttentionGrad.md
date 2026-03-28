声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# MultiScaleDeformableAttentionGrad

## 支持的产品型号

Atlas 训练系列产品/Atlas A2训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 函数原型

每个算子分为[两段式接口](./common/两段式接口.md)，必须先调用“aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMultiScaleDeformableAttentionGrad”接口执行计算。

- `aclnnStatus aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize(const aclTensor* value, const aclTensor* spatialShape, const aclTensor* levelStartIndex, const aclTensor* location, const aclTensor* attnWeight, const aclTensor* gradOutput, aclTensor* gradValue, aclTensor* gradLocation, aclTensor* gradAttnWeight, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnMultiScaleDeformableAttentionGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能说明

算子功能：
  MultiScaleDeformableAttention正向算子功能主要通过采样位置（sample location）、注意力权重（attention weights）、映射后的value特征、多尺度特征起始索引位置、多尺度特征图的空间大小（便于将采样位置由归一化的值变成绝对位置）等参数来遍历不同尺寸特征图的不同采样点。而反向算子的功能为根据正向的输入对输出的贡献及初始梯度求出输入对应的梯度。

## aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize

- **参数说明**：
  
  - value（aclTensor\*, 计算输入）：特征图的特征值，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，shape为（bs, num_keys, num_heads, channels），其中bs为batch size，num_keys为特征图的大小，num_heads为头的数量，embed_dims为特征图的维度。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - spatialShape（aclTensor\*, 计算输入）：存储每个尺度特征图的高和宽，Device侧的aclTensor，数据类型支持INT32、INT64，shape为（num_levels, 2），其中num_levels为特征图的数量，2分别代表H, W。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - levelStartIndex（aclTensor\*, 计算输入）：每张特征图的起始索引，Device侧的aclTensor，数据类型支持INT32、INT64，shape为（num_levels），其中num_levels为特征图的数量。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - location（aclTensor\*, 计算输入）：采样点位置tensor，存储每个采样点的坐标位置，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，shape为（bs, num_queries, num_heads, num_levels, num_points, 2），其中bs为batch size，num_queries为查询的数量，num_heads为头的数量，num_levels为特征图的数量，num_points为采样点的数量，2分别代表y, x。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - attnWeight（aclTensor, 计算输入）：采样点权重tensor，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，shape为（bs, num_queries, num_heads, num_levels, num_points），其中bs为batch size，num_queries为查询的数量，num_heads为头的数量，num_levels为特征图的数量，num_points为采样点的数量。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - gradOutput（aclTensor\*, 计算输入）：正向输出梯度，也是反向算子的初始梯度，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，shape为（bs, num_queries, num_heads, channels），其中bs为batch size，num_queries为查询的数量，num_heads为头的数量，embed_dims为特征图的维度。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - gradValue（aclTensor, 计算输出）：输入value对应的梯度，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，shape与value保持一致支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - gradLocation（aclTensor\*, 计算输出）：输入location对应的梯度，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，shape与location保持一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - gradAttnWeight（aclTensor\*, 计算输出）：输入attnWeight对应的梯度，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，shape与attnWeight保持一致，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**, 出参）：返回op执行器，包含了算子计算流程。
- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的输入或输出是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. 输入和输出的数据类型不在支持的范围之内。
                                        2. 输入输出数据类型不一致。
                                        3. 不满足接口约束说明章节。
  ```

## aclnnMultiScaleDeformableAttentionGrad

- **参数说明**：
  
  - workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的AscendCL Stream流。
- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](./common/aclnn返回码.md)。

## 约束说明

- 通道数channels%8 = 0，且channels<=256
- 查询的数量num_queries < 500000
- 特征图的数量num_levels <= 16
- 头的数量num_heads <= 16
- 采样点的数量num_points <= 16

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">MultiScaleDeformableAttentionGrad</td></tr>
</tr>
<tr><td rowspan="7" align="center">算子输入</td><td align="center">name</td><td align="center">type</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">value</td><td align="center">tensor</td><td align="center">bfloat16,float16,float</td><td align="center">ND</td></tr>
<tr><td align="center">spatialShape</td><td align="center">tensor</td><td align="center">int32、int64</td><td align="center">ND</td></tr>
<tr><td align="center">levelStartIndex</td><td align="center">tensor</td><td align="center">int32、int64</td><td align="center">ND</td></tr>
<tr><td align="center">location</td><td align="center">tensor</td><td align="center">bfloat16,float16,float</td><td align="center">ND</td></tr>
<tr><td align="center">attnWeight</td><td align="center">tensor</td><td align="center">bfloat16,float16,float</td><td align="center">ND</td></tr>
<tr><td align="center">gradOutput</td><td align="center">tensor</td><td align="center">bfloat16,float16,float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="4" align="center">算子输出</td><td align="center">name</td><td align="center">type</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">gradValue</td><td align="center">tensor</td><td align="center">bfloat16,float16,float</td><td align="center">ND</td></tr>
<tr><td align="center">gradLocation</td><td align="center">tensor</td><td align="center">bfloat16,float16,float</td><td align="center">ND</td></tr>
<tr><td align="center">gradAttnWeight</td><td align="center">tensor</td><td align="center">bfloat16,float16,float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">MultiScaleDeformableAttentionGrad</td></tr>
</table>

## 调用示例

详见[MultiScaleDeformableAttentionGrad自定义算子样例说明算子调用章节](../README.md#算子调用)
