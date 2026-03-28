## `CoalesceSparse`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`CoalesceSparse`算子。

### 算子描述
`CoalesceSparse`算子将相同坐标点（indices）的value进行累加求和，返回累加后的值和唯一索引。

### 算子规格描述

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

### 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品

### 目录结构介绍
```
├── docs                        // 算子文档目录
├── example                     // 调用示例目录
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用CoalesceSparse算子。</td>
    </tr>
</table>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/03/26 | 新增本readme |