## `DynamicQuantUpdateScatter`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`DynamicQuantUpdateScatter`算子。

### 算子描述
该功能为自定义算子scatter扩展功能，无对应的tensorflow或torch接口，即为DynamicQuant + Scatter+Scatter的功能组合。

### 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">DynamicQuantUpdateScatter</td></tr>
<tr><td rowspan="8" align="center">算子输入</td><td align="center">name</td><td align="center">type</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">var</td><td align="center">tensor</td><td align="center">int8</td><td align="center">ND</td></tr>

<tr><td align="center">var_scale</td><td align="center">tensor</td><td align="center">float32</td><td align="center">ND</td></tr>

<tr><td align="center">indices</td><td align="center">tensor</td><td align="center">int32, int64</td><td align="center">ND</td></tr>

<tr><td align="center">updates</td><td align="center">tensor</td><td align="center">float16, bfloat16</td><td align="center">ND</td></tr>

<tr><td align="center">smooth_scales</td><td align="center">tensor</td><td align="center">float16, bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">reduce</td><td align="center">attr</td><td align="center">string</td><td align="center">ND</td></tr>
<tr><td align="center">axis</td><td align="center">attr</td><td align="center">int</td><td align="center">ND</td></tr>

<tr><td rowspan="3" align="center">算子输出</td><td align="center">y</td><td align="center">tensor</td><td align="center">int8</td><td align="center">ND</td></tr>
<tr><td align="center">var</td><td align="center">tensor</td><td align="center">float32</td><td align="center">ND</td></tr>
<tr><td align="center">var_scale</td><td align="center">tensor</td><td align="center">float32</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">dynamic_quant_update_scatter</td></tr>
</table>

### 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品。

### 目录结构介绍
```
├── docs                        // 算子文档目录
├── example                     // 调用示例目录
├── op_host                     // host目录
├── op_kernel                   // kernel目录
├── opp_kernel_aicpu        // aicpu目录
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

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/04/27 | 新增本readme |