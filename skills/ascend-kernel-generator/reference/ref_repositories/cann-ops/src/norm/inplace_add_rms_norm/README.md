## `InplaceAddRmsNorm`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`InplaceAddRmsNorm`算子。

### 算子描述
是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。

### 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">InplaceAddRmsNorm</td></tr>
</tr>
<tr><td rowspan="5" align="center">算子输入</td><td align="center">name</td><td align="center">type</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x1</td><td align="center">tensor</td><td align="center">bfloat16,float16,float32</td><td align="center">ND</td></tr>
<tr><td align="center">x2</td><td align="center">tensor</td><td align="center">bfloat16,float16,float32</td><td align="center">ND</td></tr>
<tr><td align="center">gamma</td><td align="center">tensor</td><td align="center">bfloat16,float16,float32</td><td align="center">ND</td></tr>
<tr><td align="center">epsilon</td><td align="center">attr</td><td align="center">float32</td><td align="center">-</td></tr>
</tr>
</tr>
<tr><td rowspan="3" align="center">算子输出</td><td align="center">y</td><td align="center">tensor</td><td align="center">bfloat16,float16,float32</td><td align="center">ND</td></tr>
<tr><td align="center">rstd</td><td align="center">tensor</td><td align="center">bfloat16,float16,float32</td><td align="center">ND</td></tr>
<tr><td align="center">x</td><td align="center">tensor</td><td align="center">bfloat16,float16,float32</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">inplace_add_rms_norm</td></tr>
</table>

### 支持的产品型号
本样例支持如下产品型号：
- Atlas 训练系列产品
- Atlas 推理系列产品
- Atlas A2 训练系列产品
- Atlas 800I A2 推理产品
- Atlas 200I/500 A2 推理产品

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
    bash build.sh -n inplace_add_rms_norm
    ```

  - 部署算子包

    ```bash
    bash build_out/CANN-custom_ops-<cann_version>-linux.<arch>.run
    ```
### 算子调用
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./tests/st"> st</td><td>通过GEIR调用的方式调用InplaceAddRmsNorm算子。</td>
    </tr>
</table>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/04/06 | 新增本readme |