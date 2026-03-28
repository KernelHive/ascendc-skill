## `MaskedSelectV3`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`MaskedSelectV3`算子。

### 算子描述
`MaskedSelectV3`根据一个布尔掩码张量（mask）中的值选择输入张量（self）中的元素作为输出，形成一个新的一维张量。

### 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">MaskedSelectV3</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <!-- op::DataType::DT_FLOAT,   op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_DOUBLE, op::DataType::DT_BOOL, op::DataType::DT_BF1 -->
<tr><td align="center">x</td><td align="center">-</td><td align="center">float, int32, int64, float16, int16, int8, uint8, double, bool, bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">mask</td><td align="center">-</td><td align="center">bool</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">1D</td><td align="center">float, int32, int64, float16, int16, int8, uint8, double, bool, bfloat16</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">masked_select_v3</td></tr>
</table>

### 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品

### 目录结构介绍
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用MaskedSelectV3算子。</td>
    </tr>
</table>

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/04/10 | 新增本readme |