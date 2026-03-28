## `WeightQuantBatchMatmulV2`自定义算子样例说明 
本样例通过`Ascend C`编程语言实现了`WeightQuantBatchMatmulV2`算子。

### 算子描述
完成一个输入为伪量化场景的矩阵乘计算，并可以实现对于输出的量化计算。。
 计算公式：

  $$
  y=x @ ANTIQUANT(weight) + bias
  $$
  其中反量化公式为：
  $$
  ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
  $$
  当配置quantScaleOptional输入时，会对输出进行量化处理，其量化公式为：
  $$
  y = QUANT(x @ ANTIQUANT(weight) + bias)
  = (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset
  $$


### 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">WeightQuantBatchMatmulV2</td></tr>
</tr>
<tr><td rowspan="7" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">16 * 32</td><td align="center">float16、bfloat16</td><td align="center">ND</td></tr>
<tr><td align="center">weight</td><td align="center">32 * 16</td><td align="center">int8、int4、int32</td><td align="center">ND</td></tr>
<tr><td align="center">antiquantScale</td><td align="center">见表下方描述</td><td align="center">float16、bfloat16、uint64、int64</td><td align="center">ND</td></tr>
<tr><td align="center">antiquantOffsetOptional</td><td align="center">存在时与antiquantScale一致</td><td align="center">float16、bfloat16、int32</td><td align="center">ND</td></tr>
<tr><td align="center">quantScaleOptional</td><td align="center">见表下方描述</td><td align="center">uint64</td><td align="center">ND</td></tr>
<tr><td align="center">quantOffsetOptional</td><td align="center">存在时与quantScaleOptional一致</td><td align="center">float32</td><td align="center">ND</td></tr>

</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">16 * 16</td><td align="center">float16、bfloat16、int8</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">weight_quant_batch_matmul_v2</td></tr>
</table>

对于不同伪量化算法模式，antiquantScale支持的shape如下：

- per_tensor模式：输入shape为(1,)或(1, 1)。
- per_channel模式：输入shape为(1, n)或(n,)。
- per_group模式：输入shape为(ceil(k, group_size), n)。

对于不同的伪量化算法模式，quantScaleOptional支持的shape如下：

- per_tensor模式：输入shape为(1,)或(1, 1)。
- per_channel模式：输入shape为(1, n)或(n,)。

### 支持的产品型号
本样例支持如下产品型号：
- Atlas 训练系列产品
- Atlas 推理系列产品
- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

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
### 算子调用
<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用WeightQuantBatchMatmulV2算子。</td>
    </tr>
</table>

### 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/04/07 | 新增本readme |
