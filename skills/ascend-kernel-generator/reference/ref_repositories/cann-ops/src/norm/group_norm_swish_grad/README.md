## `GroupNormSwishGrad`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`GroupNormSwishGrad`算子。

### 算子描述
`GroupNormSwishGrad`算子实现GroupNorm与Swish融合算子的反向计算。

### 算子规格描述

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">CrossEntropyLossGrad</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="7" align="center">算子输入</td>

<tr><td align="center">dy</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  

<tr><td align="center">mean</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr> 

<tr><td align="center">rstd</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr> 

<tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>

<tr><td align="center">gamma</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>

<tr><td align="center">beta</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>

<tr><td rowspan="4" align="center">算子输出</td>

<tr><td align="center">dxOut</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>

<tr><td align="center">dgammaOutOptional</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>

<tr><td align="center">dbetaOutOptional</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>

<tr><td rowspan="6" align="center">算子属性</td>
<tr><td align="center">numGroups</td><td align="center">scalar</td><td align="center">int</td><td align="center">-</td></tr>
<tr><td align="center">dataFormat</td><td align="center">scalar</td><td align="center">char*</td><td align="center">-</td></tr>
<tr><td align="center">swishScale</td><td align="center">scalar</td><td align="center">float</td><td align="center">-</td></tr>
<tr><td align="center">dgammaIsRequire</td><td align="center">scalar</td><td align="center">bool</td><td align="center">-</td></tr>
<tr><td align="center">dbetaIsRequire</td><td align="center">scalar</td><td align="center">bool</td><td align="center">-</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">GroupNormSwishGrad</td></tr>
</table>

### 支持的产品型号
本样例支持如下产品型号：
- Atlas 训练系列产品
- Atlas 推理系列产品
- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

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
    cd ${git_clone_path}/ops-contribution
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用GroupNormSwishGrad算子。</td>
    </tr>
</table>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/04/23 | 新增本readme |