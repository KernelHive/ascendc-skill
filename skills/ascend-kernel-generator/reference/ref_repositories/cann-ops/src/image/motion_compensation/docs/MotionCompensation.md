声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# MotionCompensation

## 支持的产品型号

Atlas 推理系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 算子功能：
对输入点云中的每个点，根据其时间戳在 [timestamp_min, timestamp_max] 区间内的相对位置，对整体平移和旋转进行线性插值，从而把该点从“最小位姿坐标系”变换到“最大位姿坐标系”，实现运动畸变补偿。

- 计算公式：

  1. 计算时间比例因子  
     $$ t = \frac{\text{timestamp\_max} - \text{timestamp}_i}{\text{timestamp\_max} - \text{timestamp\_min}} \quad (t \in [0, 1]) $$

  2. 计算全局相对平移  
     $$ \mathbf{T}_{\text{global}} = \mathbf{t}_{\text{min}} - \mathbf{t}_{\text{max}} $$

  3. 将全局平移旋转到最大时间坐标系  
     $$ \mathbf{T} = \mathbf{R}(\mathbf{q}_{\text{max}}^\dagger) \cdot \mathbf{T}_{\text{global}} $$  
     其中 $\mathbf{q}_{\text{max}}^\dagger$ 为 $\mathbf{q}_{\text{max}}$ 的共轭四元数。

  4. 计算相对旋转四元数  
     $$ \mathbf{q}_{\text{rel}} = \mathbf{q}_{\text{max}}^\dagger \otimes \mathbf{q}_{\text{min}} $$  
     并归一化：$\mathbf{q}_{\text{rel}} \leftarrow \frac{\mathbf{q}_{\text{rel}}}{\|\mathbf{q}_{\text{rel}}\|}$。

  5. 判断是否需要旋转补偿  
     $$ \text{do\_rotation} = |\mathbf{q}_{\text{rel}} \cdot \mathbf{q}_{\text{identity}}| < 1 - \varepsilon $$  
     若成立，则计算球面线性插值 (SLERP) 系数：
     $$ \theta = \arccos(|\mathbf{q}_{\text{rel}} \cdot \mathbf{q}_{\text{identity}}|) $$  
     $$ \mathbf{q}_{\text{interp}} = \frac{\sin((1 - t)\theta)}{\sin\theta}\mathbf{q}_{\text{identity}} + \frac{\sin(t\theta)}{\sin\theta}\, \text{sgn}(\mathbf{q}_{\text{rel}} \cdot \mathbf{q}_{\text{identity}})\cdot \mathbf{q}_{\text{rel}} $$

  6. 补偿后的点坐标  
     $$ \mathbf{p}_{\text{out}} = \mathbf{R}(\mathbf{q}_{\text{interp}})\cdot \mathbf{p}_{\text{in}} + t\cdot \mathbf{T} $$  
     若 `do_rotation` 为假，则仅平移：  
     $$ \mathbf{p}_{\text{out}} = \mathbf{p}_{\text{in}} + t\cdot \mathbf{T} $$

- 边界/特殊处理：
  - 当 `timestamp_max == timestamp_min` 时，所有点直接返回原始坐标。
  - 如果某点坐标包含 NaN，则直接复制该点到输出，不做补偿。
  - 输入点云维度 `ndim` 可以大于 3；第 4 维及以后（例如强度）原样拷贝到输出。

## 实现原理

MotionCompensation由向量API组合实现。

## 算子执行接口

每个算子分为[两段式接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E4%B8%A4%E6%AE%B5%E5%BC%8F%E6%8E%A5%E5%8F%A3.md)，必须先调用“aclnnMotionCompensationGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMotionCompensation”接口执行计算。

* `aclnnStatus aclnnMotionCompensationGetWorkspaceSize(const aclTensor *input, const aclTensor *other, const int64_t timestampMin, const int64_t timestampMax, const aclFloatArray *transMin, const aclFloatArray *transMax, const aclFloatArray *qMin, const aclFloatArray *qMax, const aclTensor *out, uint64_t workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMotionCompensation(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnMotionCompensationGetWorkspaceSize

- **参数说明：**
  
  - input（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入$p_{in}$，数据类型支持Float32，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - other（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入timestamp，数据类型支持UINT64，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - timestampMin（int64，计算输入）：必选参数，Host侧的int64，公式中的输入$timestamp_{min}$，数据类型支持INT64。
  - timestampMax（int64，计算输入）：必选参数，Host侧的int64，公式中的输入$timestamp_{max}$，数据类型支持INT64。
  - tranMin（aclFloatArray\*，计算输入）：必选参数，Device侧的aclFloatArray，数据类型支持Float32。
  - tranMax（aclFloatArray\*，计算输入）：必选参数，Device侧的aclFloatArray，数据类型支持Float32。
  - qMin（aclFloatArray\*，计算输入）：必选参数，Device侧的aclFloatArray，数据类型支持Float32。
  - qMax（aclFloatArray\*，计算输入）：必选参数，Device侧的aclFloatArray，数据类型支持Float32。
  - out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出$p_{out}$，数据类型支持Float32，[数据格式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)支持ND。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：input、other、out、timestampMin、timestampMax、transMin、transMax、qMin、qMax的数据类型和数据格式不在支持的范围内。
  ```

### aclnnMotionCompensation

- **参数说明：**
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMotionCompensationGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- input，out的数据类型只支持Float32，数据格式只支持ND
- other的数据类型只支持UINT64，数据格式只支持ND

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">MotionCompensation</td></tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">points</td><td align="center">ndim, N</td><td align="center">float32</td><td align="center">ND</td></tr>
<tr><td align="center">timestamps</td><td align="center">N</td><td align="center">uint64</td><td align="center">ND</td></tr>
<tr><td rowspan="6" align="center">算子属性</td><td align="center">timestamp_min</td><td align="center">1</td><td align="center">int64</td><td align="center">ND</td></tr>
<tr><td align="center">timestamp_max</td><td align="center">1</td><td align="center">list_float</td><td align="center">ND</td></tr>
<tr><td align="center">translation_min</td><td align="center">3</td><td align="center">list_float</td><td align="center">ND</td></tr>
<tr><td align="center">translation_max</td><td align="center">3</td><td align="center">list_float</td><td align="center">ND</td></tr>
<tr><td align="center">quaternion_min</td><td align="center">4</td><td align="center">list_float</td><td align="center">ND</td></tr>
<tr><td align="center">quaternion_max</td><td align="center">4</td><td align="center">list_float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center">ndim, N</td><td align="center">float32</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">motion_compensation</td></tr>
</table>

## 调用示例

详见[MotionCompensation自定义算子样例说明算子调用章节](../README.md#算子调用)
