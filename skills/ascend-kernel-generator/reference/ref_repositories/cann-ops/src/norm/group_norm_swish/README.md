## `GroupNormSwish`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`GroupNormSwish`算子。

### 算子描述
`GroupNormSwish`算子计算输入x的组归一化结果out，均值meanOut，标准差的倒数rstdOut，以及swish的输出。

### 算子规格描述

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">GroupNormSwish</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="4" align="center">算子输入</td>

<tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  

<tr><td align="center">gamma</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr> 

<tr><td align="center">beta</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr> 

<tr><td rowspan="3" align="center">算子输出</td>

<td align="center">yOut</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<td align="center">meanOut</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<td align="center">rstdOut</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>

<tr><td rowspan="5" align="center">算子属性</td>
<td align="center">numGroups</td><td align="center">scalar</td><td align="center">int64_t</td><td align="center">-</td></tr>
<td align="center">dataFormatOptional</td><td align="center">scalar</td><td align="center">char*</td><td align="center">-</td></tr>
<td align="center">eps</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>
<td align="center">activateSwish</td><td align="center">scalar</td><td align="center">bool</td><td align="center">-</td></tr>
<td align="center">swishScale</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">GroupNormSwish</td></tr>
</table>

### 支持的产品型号
本样例支持如下产品型号：
- 昇腾910B AI处理器

### 目录结构介绍
```
├── docs                        // 算子文档目录
├── example                     // 调用示例目录
├── op_host                     // host目录
├── op_kernel                   // kernel目录
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用GroupNormSwish算子。</td>
    </tr>

</table>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/04/23 | 新增本readme |
