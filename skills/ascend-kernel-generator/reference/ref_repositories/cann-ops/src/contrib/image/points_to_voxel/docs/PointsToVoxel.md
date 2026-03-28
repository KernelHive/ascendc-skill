声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# PointsToVoxel

## 支持的产品型号

- Atlas 推理系列产品
- Atlas 800l A2推理产品
- Atlas A2训练系列产品


产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。


## 功能描述

- 算子功能：PointsToVoxel算子将3D空间中的点云划分成规则的体素网格（Voxel Grid），并统计每个体素内的点云信息。


## 实现原理

PointsToVoxel由向量API组合实现。
  1. 在Host侧获取输入点云的总点数，计算每个维度上的体素网格数量，将网格尺寸转为整数
  2. 在Kernel调用Ascend C的API接口Sub、Div、Cast、Min、Mins、Mul，计算每个点在体素网格中的坐标，判断是否在网格范围内
  3. 如果点有效，检查它是否属于新的体素，如果体素未被占用，则分配新的体素索引。如果体素数量大于最大值，则跳过该点，否则将结果写入coors_out。如果体素已存在，直接获取它的索引
  4. 检查当前体素的点数，如果点数小于max_points，将该点存入voxels_out，并增加计数 num_points_per_voxel[voxelidx] += 1
  5. 实际生成的体素数量写入voxel_num

## 算子执行接口

每个算子分为两段式接口，必须先调用“aclnnPointsToVoxelGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnPointsToVoxel”接口执行计算。

* `aclnnStatus aclnnPointsToVoxelGetWorkspaceSize(const aclTensor* points, const aclFloatArray *voxel_size, const aclFloatArray *coors_range, const int32_t max_points, const bool reverse_index, const int32_t max_voxels, const aclTensor* voxels_out, const aclTensor* coors_out, const aclTensor* num_points_per_voxel, const aclTensor* voxel_num, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnPointsToVoxel(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnPointsToVoxelGetWorkspaceSize

- **参数说明：**

  - points（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入points，数据类型支持FLOAT32，数据格式支持ND，shape为[ndim, N]
  - voxel_size（aclFloatArray\*，算子属性）：Device侧的aclFloatArray，数据类型支持FLOAT32，数组的长度为3
  - coors_range（aclFloatArray\*，算子属性）：Device侧的aclFloatArray，数据类型支持FLOAT32，数组的长度为6
  - max_points（int，算子属性）：Host侧的int，支持数据类型为int
  - reverse_index（bool，算子属性）：Host侧的bool，支持数据类型为bool  
  - max_voxels（int，算子属性）：Host侧的int，支持数据类型为int  
  - voxels_out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出voxels_out，数据类型支持FLOAT32，数据格式支持ND，shape为[voxel_num, max_points, ndim]
  - coors_out（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出coors_out，数据类型支持INT32，数据格式支持ND，shape为[voxel_num, 3]
  - num_points_per_voxel（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出num_points_per_voxel，数据类型支持INT32，数据格式支持ND，shape为[voxel_num]
  - voxel_num（aclTensor\*，计算输出）：Device侧的aclTensor，公式中的输出voxel_num，数据类型支持INT32，数据格式支持ND，shape为[1]
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。



- **返回值：**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。
  
  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：points、voxel_size、coors_range、max_points、reverse_index、max_voxels、voxels_out、coors_out、num_points_per_voxel、voxel_num的数据类型和数据格式不在支持的范围内。
    ```

### aclnnPointsToVoxel

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存起址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnFFNGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。
- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/common/aclnn%E8%BF%94%E5%9B%9E%E7%A0%81_fuse.md)。

## 约束与限制

- points的数据类型支持FLOAT32，数据格式只支持ND
- points[:3, :] 包含 xyz 坐标，points[3:, :] 包含其他信息
- 算子属性coors_range表示体素范围，格式：xyzxyz，即最小最大值
- 算子属性max_points表示一个体素中最大包含的点数
- 算子属性reverse_index表示是否返回反转坐标。如果points是xyz格式，且reverse_index为True，输出坐标将为zyx格式，但特征中的点始终为xyz格式
- 算子属性max_voxels表示此函数创建的最大体素数


## 算子原型

<table>  
  <tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">PointsToVoxel</th></tr>  
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">Type</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>  
  <tr><td align="center">points</td><td align="center">tensor</td><td align="center">[ndim, N]</td><td align="center">float32</td><td align="center">ND</td></tr>    
  <tr>
    <td rowspan="4" align="center">算子输出</td>
    <td align="center">voxels_out</td><td align="center">tensor</td><td align="center">[voxel_num, max_points, ndim]</td><td align="center">float32</td><td align="center">ND</td></tr>
    <td align="center">coors_out</td><td align="center">tensor</td><td align="center">[voxel_num, 3]</td><td align="center">int32</td><td align="center">ND</td></tr>
    <td align="center">num_points_per_voxel</td><td align="center">tensor</td><td align="center">[voxel_num]</td><td align="center">int32</td><td align="center">ND</td></tr>
    <td align="center">voxel_num</td><td align="center">tensor</td><td align="center">[1]</td><td align="center">int32</td><td align="center">ND</td>
  </tr>
  <tr>
    <td rowspan="5" align="center">算子属性</td>
    <td align="center">voxel_size</td><td align="center">ListFloat</td><td align="center">[3]</td><td align="center">NA</td><td align="center">NA</td></tr>
    <td align="center">coors_range</td><td align="center">ListFloat</td><td align="center">[6]</td><td align="center">NA</td><td align="center">NA</td></tr>
    <td align="center">max_points</td><td align="center">int</td><td align="center">NA</td><td align="center">NA</td><td align="center">NA</td></tr>
    <td align="center">reverse_index</td><td align="center">bool</td><td align="center">NA</td><td align="center">NA</td><td align="center">NA</td></tr>  
    <td align="center">max_voxels</td><td align="center">int</td><td align="center">NA</td><td align="center">NA</td><td align="center">NA</td></tr> 
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">points_to_voxel</td></tr>
</table>

## 调用示例

详见[PointsToVoxel自定义算子样例说明算子调用章节](../README.md#算子调用)