## `MotionCompensation`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`MotionCompensation`算子。

### 算子描述
`MotionCompensation`算子根据给定的最小/最大时间戳、最小/最大位姿（平移 + 四元数），对一组带有时间戳的三维点云进行运动畸变补偿，输出补偿后的点云坐标。

### 算子规格描述

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

### 支持的产品型号
本样例支持如下产品型号：
- Atlas 推理系列产品

### 目录结构介绍
```
├── docs                        // 算子文档目录
├── example                     // 调用示例目录
├── framework                   // 第三方框架适配目录
├── op_host                     // host目录
├── op_kernel                   // kernel目录
├── opp_kernel_aicpu            // aicpu目录
└── tests                       // 测试用例目录
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用MotionCompensation算子。</td>
    </tr>
</table>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/07/26 | 新增本readme |
