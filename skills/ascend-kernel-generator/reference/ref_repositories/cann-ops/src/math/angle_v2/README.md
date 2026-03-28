## AngleV2自定义算子样例说明 
本样例通过Ascend C编程语言实现了AngleV2算子，并按照不同的算子调用方式分别给出了对应的端到端实现。

## 算子描述
计算给定输入张量的幅角（以弧度为单位）

- 当输入x为复数时，其中x.real和x.imag分别代表x的实部和虚部
  $$y = \left\{
    \begin{array}{rcl}
    &atan(\frac{x.imag}{x.real}) & x.real \gt 0 \\
    &\frac{\pi}{2} & x.real = 0 \&\& x.imag \ge 0 \\
    &0 & x.real = 0 \&\& x.imag = 0 \\
    &-\frac{\pi}{2} & x.real = 0 \&\& x.imag \lt 0 \\
    &\pi + atan(\frac{x.imag}{x.real}) & x.real \lt 0 \&\& x.imag \ge 0 \\
    &-\pi + atan(\frac{x.imag}{x.real}) & x.real \lt 0 \&\& x.imag \lt 0\\
    &sign(x.imag) \times \pi & x.real = -\inf \&\& 0 \lt abs(x.imag) \lt \inf\\
    &0 & x.real = \inf \&\& 0 \lt abs(x.imag) \lt \inf\\
    &sign(x.imag) \times \frac{\pi}{4} & x.real = \inf \&\& abs(x.imag) = \inf\\
    &sign(x.imag) \times \frac{3\pi}{4} & x.real = -\inf \&\& abs(x.imag) = \inf\\
    &sign(x.imag) \times \frac{\pi}{2} & abs(x.real) \lt \inf \&\& abs(x.imag) = \inf\\
    \end{array}
    \right.
  $$

- 当输入x为实数时
  $$y = \left\{
    \begin{array}{rcl}
    &0 & x \ge 0 \\
    &\pi & x \lt 0 \\
    \end{array}
    \right.
  $$


## 算子规格描述
<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">AngleV2</th></tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">默认值</td></tr>

<tr><td align="center">x</td><td align="center">-</td><td align="center">float32, float16, complex64, bool, uint8, int8, int16, int32, int64</td><td align="center">ND</td><td align="center">\</td></tr>

<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">-</td><td align="center">float32, float16</td><td align="center">ND</td><td align="center">\</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">angle_v2</td></td></tr>
</table>


## 支持的产品型号
本样例支持如下产品型号：
- Atlas 训练系列产品
- Atlas A2训练系列产品

### 目录结构介绍
```
├── docs                        // 算子文档目录
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
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用AngleV2算子。</td>
    </tr>
</table>

## 更新说明
| 时间 | 更新事项 |
|----|------|
| 2025/03/26 | 新增本readme |