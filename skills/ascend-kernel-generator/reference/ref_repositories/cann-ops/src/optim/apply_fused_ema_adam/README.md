## `ApplyFusedEmaAdam`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`ApplyFusedEmaAdam`算子。

### 算子描述
- **算子功能**：实现FusedEmaAdam融合优化器功能。
- **计算公式**：

  $$
  (correction_{\beta_1},correction_{\beta_2},)=\begin{cases}
  (1,1),&biasCorrection=False\\
  (1-\beta_1^{step},1-\beta_2^{step}),&biasCorrection=True
  \end{cases}
  $$
  
  $$
  grad=\begin{cases}
  grad+weightDecay*var,&mode=0\\
  grad,&mode=1
  \end{cases}
  $$
  
  $$
  m_{out}=\beta_1*m+(1-\beta_1)*grad
  $$

  $$
  v_{out}=\beta_2*v+(1-\beta_2)*grad^2
  $$

  $$
  m_{next}=m_{out}/correction_{\beta_1}
  $$

  $$
  v_{next}=v_{out}/correction_{\beta_2}
  $$

  $$
  denom=\sqrt{v_{next}}+eps
  $$

  $$
  update=\begin{cases}
  m_{next}/denom,&mode=0\\
  m_{next}/denom+weightDecay*var,&mode=1
  \end{cases}
  $$

  $$
  var_{out}=var-lr*update
  $$

  $$
  s_{out}=emaDecay*s+(1-emaDecay)*var_{out}
  $$

### 算子规格描述

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">ApplyAdamWV2</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="2" align="center">算子输入/输出</td> 
<tr><td align="center">grad</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入/输出</td> 
<tr><td align="center">varRef</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入/输出</td>
<tr><td align="center">mRef</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入/输出</td>
<tr><td align="center">vRef</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入/输出</td>
<tr><td align="center">sRef</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">算子输入</td>
<tr><td align="center">step</td><td align="center">tensor</td><td align="center">int64</td><td align="center">ND</td></tr>  

<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">lr</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">emaDecay</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">beta1</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">beta2</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">eps</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">mode</td><td align="center">scalar</td><td align="center">int64</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">biasCorrection</td><td align="center">scalar</td><td align="center">bool</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">算子属性</td>
<td align="center">weightDecay</td><td align="center">scalar</td><td align="center">double</td><td align="center">-</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">apply_fused_ema_adam</td></tr>  
</table>


### 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品
- Atlas 200I/500 A2推理产品


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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用ApplyFusedEmaAdam算子。</td>
    </tr>
</table>

## 更新说明
| 时间         | 更新事项 |
|------------|------|
| 2025/03/25 | 新增本readme |