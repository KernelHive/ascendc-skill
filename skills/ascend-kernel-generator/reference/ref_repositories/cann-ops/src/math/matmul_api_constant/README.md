## `MatmulApiConstant`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`MatmulApiConstant`算子。

### 算子描述
MatmulApiConstant算子在调用Matmul API实现矩阵乘法时，通过使用全量常量化的MatmulApiStaticTiling模板参数，替代非常量的TCubeTiling参数。将计算提前到编译期，以减少运行时的Scalar计算开销，实现算子性能提升。
适用场景：
  - Matmul初始化时的Scalar计算较多，影响指令头开销。
  - Matmul迭代之间的Scalar计算较多，阻塞MTE2流水。
限制：需要在编译期就确定部分Tiling参数，需满足以下条件之一：
  - 全量常量化：可以确定的Single Shape（SingleCoreM/SingleCoreN/SingelCoreK）和Base Shape（baseM/baseN/baseK）。
  - 部分常量化：可以确定的Base Shape（baseM/baseN/baseK）。
其中全量常量化场景比部分常量化场景减少更多的Scalar计算开销。

### 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">MatmulApiConstant</td></tr>
</tr>
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">bias</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_api_constant</td></tr>
</table>

### 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品/Atlas 800I A2推理产品

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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用MatmulApiConstant算子。</td>
    </tr>
</table>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/05/02 | 新增本readme |
