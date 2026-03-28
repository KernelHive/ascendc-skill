## `Muls`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`Muls`算子。

### 算子描述
`Muls`算子用将输入张量Input乘标量value，将结果返回到输出张量。

计算公式为：
$$
output = Input * value
$$

### 算子规格描述

<table border="1">
  <tr>
    <th align="center" rowspan="2">算子类型(OpType)</th>
    <th colspan="4" align="center">Muls</th>
  </tr>
  <tr>
    <th align="center">name</th>
    <th align="center">Type</th>
    <th align="center">data type</th>
    <th align="center">format</th>
  </tr>
     <tr>
    <td rowspan="1" align="center">算子输入</td>
    <td align="center">Input</td>
    <td align="center">tensor</td>
    <td align="center">float32, float16，bf16，int16,int32, int64,complex64</td>
    <td align="center">ND</td>
  </tr>
    <tr>
    <td rowspan="1" align="center">标量</td>
    <td align="center">value</td>
    <td align="center">Scalar</td>
    <td align="center">float</td>
    <td align="center">ND</td>
  </tr>
   <tr>
    <td rowspan="1" align="center">算子输出</td>
    <td align="center">output</td>
    <td align="center">tensor</td>
    <td align="center">float32, float16，bf16，int16,int32, int64,complex64</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td rowspan="1" align="center">核函数名</td>
    <td colspan="4" align="center">muls</td>
  </tr>
</table>
 


### 支持的产品型号
本样例支持如下产品型号：
- Atlas A2 训练系列产品
- Atlas 800I A2推理产品


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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Muls算子。</td>
    </tr>
</table>

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/03/30 | 新增本readme |