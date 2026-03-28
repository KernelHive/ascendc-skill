## Conv2DBackpropFilterV3自定义算子样例说明 
本样例通过Ascend C编程语言实现了Conv2DBackpropFilterV3算子，并按照不同的算子调用方式分别给出了对应的端到端实现。


## 算子规格描述
<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">Conv2DBackpropFilterV3</th></tr>
<tr><td rowspan="9" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">默认值</td></tr>

<tr><td align="center">x</td><td align="center">-</td><td align="center">float32, float16, bfloat16</td><td align="center">NC1HWC0</td><td align="center">\</td></tr>

<tr><td align="center">filter_size</td><td align="center">-</td><td align="center">int32</td><td align="center">ND</td><td align="center">\</td></tr>

<tr><td align="center">out_backprop</td><td align="center">-</td><td align="center">float32, float16, bfloat16</td><td align="center">NC1HWC0</td><td align="center">\</td></tr>

<tr><td align="center">stride</td><td align="center">list</td><td align="center">int64</td><td align="center">-</td><td align="center">\</td></tr>

<tr><td align="center">pads</td><td align="center">list</td><td align="center">int64</td><td align="center">-</td><td align="center">\</td></tr>

<tr><td align="center">dilations</td><td align="center">list</td><td align="center">int64</td><td align="center">-</td><td align="center">{1,1,1,1}</td></tr>

<tr><td align="center">groups</td><td align="center">-</td><td align="center">int64</td><td align="center">-</td><td align="center">1</td></tr>

<tr><td align="center">data_format</td><td align="center">-</td><td align="center">string</td><td align="center">-</td><td align="center">NCHW</td></tr>


<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">-</td><td align="center">float32</td><td align="center">FRACTAL_Z</td><td align="center">\</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">conv2d_backprop_filter_v3</td></td></tr>
</table>


## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品

### 目录结构介绍
```
├── example                     // 调用示例目录
├── op_host                     // host目录
├── op_kernel                   // kernel目录
├── opp_kernel_aicpu            // aicpu目录
└── tests                       // 测试用例目录
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用Conv2DBackpropFilterV3算子。</td>
    </tr>
</table>

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/04/30 | 新增本readme |