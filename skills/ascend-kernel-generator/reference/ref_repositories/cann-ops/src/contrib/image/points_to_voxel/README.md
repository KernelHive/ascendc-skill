## `PointsToVoxel`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`PointsToVoxel`算子。

### 算子描述
PointsToVoxel算子将3D空间中的点云划分成规则的体素网格（Voxel Grid），并统计每个体素内的点云信息


### 算子规格描述

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


### 支持的产品型号
本样例支持如下产品型号：
- Atlas 推理系列产品
- Atlas 800l A2推理产品
- Atlas A2训练系列产品



### 目录结构介绍
```
├── docs                        // 算子文档目录
├── example                     // 调用示例目录
├── framework                   // 第三方框架适配目录
├── op_host                     // host目录
├── op_kernel                   // kernel目录
├── opp_kernel_aicpu            // aicpu目录
└── tests                       // 测试用例目录
```


### 环境要求
编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

### 算子包编译部署
  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}/cann-ops
    ```

  - 执行编译

    ```bash
    bash build.sh
    ```

  - 部署算子包

    ```bash
    bash build_out/CANN-custom_ops-<cann_version>-linux.<arch>.run
    ```

### 算子调用
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用PointsToVoxel算子。</td>
    </tr>
</table>

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/07/24 | 新增本readme |