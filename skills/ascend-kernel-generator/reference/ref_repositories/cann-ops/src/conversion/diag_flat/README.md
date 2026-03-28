## `DiagFlat`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`DiagFlat`算子。

### 算子描述
`DiagFlat`算子：如果input是向量（一维向量），则返回二维矩阵张量，其中input元素为对角线；如果input是二维及以上的张量，则先进行扁平化，化简为一维向量，在转化为第一种场景进行出处理。

### 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">DiagFlat</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">self</td><td align="center">1-8 维度</td><td align="center">详情见docs</td><td align="center">ND</td></tr>
<tr><td align="center">diagonal</td><td align="center">1-8 维度</td><td align="center">详情见docs</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center"></td><td align="center">详情见docs</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">diag_flat</td></tr>
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用DiagFlat算子。</td>
    </tr>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/06/04 | 新增本readme |
