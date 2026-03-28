## `ClipByValueV2`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`ClipByValueV2`算子。

### 算子描述
`ClipByValueV2`算子用于将一个张量值剪切到指定的最小值和最大值之间。

### 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">ClipByValueV2</td></tr>
</tr>
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">7, 2045</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">clip_value_min</td><td align="center">1</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">clip_value_max</td><td align="center">1</td><td align="center">float16</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">7, 2045</td><td align="center">float16</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">clip_by_value_v2</td></tr>
</table>

### 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品
- Atlas 800I A2推理产品

### 目录结构介绍
```
├── docs                        // 算子文档目录
├── example                     // 调用示例目录
├── framework                   // 第三方框架适配目录
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用ClipByValueV2算子。</td>
    </tr>
</table>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/07/03 | 新增本readme |