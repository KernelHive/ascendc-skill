## `CTCLossV3Grad`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`CTCLossV3Grad`算子。

### 算子描述
`CTCLossV3Grad`算子对log_probs等输入进行梯度计算，返回log_probs梯度。

## 算子规格描述

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">CTCLossV3Grad</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="8" align="center">算子输入</td>

<tr><td align="center">grad_out</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>  

<tr><td align="center">log_probs</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr> 

<tr><td align="center">targets</td><td align="center">tensor</td><td align="center">int32,int64</td><td align="center">ND</td></tr> 

<tr><td align="center">input_lengths</td><td align="center">tensor</td><td align="center">int32,int64</td><td align="center">ND</td></tr>

<tr><td align="center">target_lengths</td><td align="center">tensor</td><td align="center">int32,int64</td><td align="center">ND</td></tr>

<tr><td align="center">neg_log_likely_hood</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>

<tr><td align="center">log_alpha</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>

<tr><td rowspan="1" align="center">算子输出</td>

<td align="center">grad</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>

<tr><td rowspan="3" align="center">算子属性</td>
<td align="center">blank</td><td align="center">attr</td><td align="center">int</td><td align="center">-</td></tr>
<td align="center">reduction</td><td align="center">attr</td><td align="center">string</td><td align="center">-</td></tr>
<td align="center">zero_infinity</td><td align="center">attr</td><td align="center">bool</td><td align="center">-</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">ctc_loss_v3_grad</td></tr>
</table>

## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2 训练系列产品

## 目录结构介绍
```
├── docs                        // 算子文档目录
├── example                     // 调用示例目录
├── op_host                     // host目录
├── op_kernel                   // kernel目录
├── opp_kernel_aicpu            // aicpu目录
└── tests                       // 测试用例目录
```

## 环境要求
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用CTCLossV3Grad算子。</td>
    </tr>
</table>

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/03/24 | 新增本readme |