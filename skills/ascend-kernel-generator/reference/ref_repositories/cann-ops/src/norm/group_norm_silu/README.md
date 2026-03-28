## `GroupNormSilu`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`GroupNormSilu`算子。

### 算子描述
- **算子功能**：计算输入self的组归一化结果out，均值meanOut，标准差的倒数rstdOut，以及silu的输出。。
- **计算公式**：
  记 $x=self$, $E[x] = \bar{x}$代表$x$的均值，$Var[x] = \frac{1}{n - 1} * \sum_{i=1}^n(x_i - E[x])^2$代表$x$的样本方差，则
  $$
  \left\{
  \begin{array} {rcl}
  out& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
  meanOut& &= E[x]\\
  rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
  \end{array}
  \right.
  $$
### 算子规格描述

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">GroupNormSilu</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入</td> 
<tr><td align="center">self</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入</td> 
<tr><td align="center">gamma</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">beta</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输出</td>
<tr><td align="center">out</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输出</td>
<tr><td align="center">meanOut</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输出</td>
<tr><td align="center">rstdOut</td><td align="center">tensor</td><td align="center">int64</td><td align="center">ND</td></tr>  

<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">group</td><td align="center">scalar</td><td align="center">int64</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">eps</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">activateSilu</td><td align="center">scalar</td><td align="center">bool</td><td align="center">-</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">group_norm_silu</td></tr>  
</table>


### 支持的产品型号
本样例支持如下产品型号：
- Atlas A2 训练系列产品
- Atlas 200I/500 A2推理产品
- Atlas 推理系列产品


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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用GroupNormSilu算子。</td>
    </tr>
</table>

## 更新说明
| 时间         | 更新事项 |
|------------|------|
| 2025/03/25 | 新增本readme |