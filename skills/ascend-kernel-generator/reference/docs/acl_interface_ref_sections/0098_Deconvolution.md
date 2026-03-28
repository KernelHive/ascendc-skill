### Deconvolution

## 总体约束

设：
- `Ny, Cy, Hy, Wy [x1] = x.shape`
- `Nk, Ck/group, Hk, Wk = filter.shape` (k=kernel)

`stride`、`stride_h`、`stride_w` 和 `pad`、`pad_h`、`pad_w` 处理逻辑如下：

```python
if stride.size() == 0:
    stride_h, stride_w
elif stride.size() == 1:
    stride_h, stride_w = stride[0]
else:
    stride_h = stride[0]
    stride_w = stride[1]

if pad.size() == 0:
    pad_h, pad_w
elif pad.size() == 1:
    pad_h = pad_w = pad[0]
else:
    pad_h = pad[0]
    pad_w = pad[1]

if dilation.size() == 1:
    dilation_h = dilation_w = dilation[0]
else:
    dilation_h = dilation[0]
    dilation_w = dilation[1]
```

设输出 shape 为 `[Nx, Cx, Hx, Wx]`，其中：

- `Nx = Ny`, `Cx = Ck`
- `Wx = (Wy - 1) * stride_w - 2 * pad_w + dilation_h * (Wk - 1) + 1`
- `Hx = (Hy - 1) * stride_h - 2 * pad_h + dilation_w * (Hk - 1) + 1`

则总体限制如下：

1. `1 <= stride_h * stride_w < 256`
2. `Hy + 2 * pad_h >= dilation_h * (Hk - 1) + 1`, `Wy + 2 * pad_w >= dilation_w * (Wk - 1) + 1`
3. `Hy * stride_h`, `Wy * stride_w` 范围为 `[2, 4096]`：
   - 当 `Hk == Wk == 1` 时：
     - 限制加强为：`Wy * stride_w * stride_h` 为 `[2, 4096]`，并且当 `stride_h > 1` 或 `stride_w > 1` 时：`Wy <= 2032`
   - 当 Atlas 训练系列产品场景时：`Hx != Hk`, `Wx != Wk` 支持 `Hy` 或 `Wy` 为 1
   - 当 `Hy = 1`, `Hk = 1`, `pad_h = 0`, `stride_h = 1`, `dilation_h = 1` 时，支持 `Wx` 范围为 `[1, 2147483647]`（int32 可表示的范围）
4. AL1Size 限制计算方式如下：

   设 `w_value = Wy * stride_w`

   ```python
   if Wx > 16:
       h_value_max = Hk + 1
   elif 16 % Wx == 0:
       h_value_max = Hk + 16 // Wx - 1
   else:
       h_value_max = Hk + 16 // Wx + 1
   ```

   则：
   - `AL1Size = h_value_max * w_value * 32Byte`
   - `BL1Size = Hk * Wk * 512Byte`
   - `AL1Size + BL1Size <= l1_size`（`l1_size` 见对应芯片的配置文件，例如 `platform_config/${soc_version}.ini`）

   其中：
   - `Hy >= dilation * (Hk - 1) + 1`
   - `Wy >= dilation * (Wk - 1) + 1`

   说明：当 `Hy = 1`, `Hk = 1`, `pad_h = 0`, `stride_h = 1`, `dilation_h = 1` 时，`AL1Size = [15 * stride_w + (Wk - 1) * dilation_w + 1] * 32B`

---

## 输入

### x

| 字段 | 说明 |
|------|------|
| 是否必填 | 必填 |
| 数据类型 | float16 |
| 参数解释 | 输入 Tensor |
| 规格限制 | 无 |

### filter

| 字段 | 说明 |
|------|------|
| 是否必填 | 必填 |
| 数据类型 | float16 |
| 参数解释 | 卷积核，shape 为 `[Nk, Ck, Hk, Wk]`，`Nk` 必须与 `Cy` 相等 |
| 规格限制 | <ul><li>如果 `Wx == Hx == 1`：`Wy == Hy == Wk == Hk`，且取值范围在 `1~11`</li><li>不支持输出的 `W == 1`，输出的 `H` 不等于 1 的场景</li><li>其它场景：`Wx` 和 `Hx` 配置范围在 `1~255`</li></ul> |

### bias

| 字段 | 说明 |
|------|------|
| 是否必填 | 非必填 |
| 数据类型 | float16 |
| 参数解释 | 无 |
| 规格限制 | 无 |

---

## 属性

### pad

| 字段 | 说明 |
|------|------|
| 是否必填 | 非必填 |
| 数据类型 | ListInt |
| 参数解释 | 高和宽轴开始和结束的 Padding，`pad` 和 `pad_h`/`pad_w` 不能同时提供，默认值为 0；`pad` List 的长度最大为 2 |
| 规格限制 | <ul><li>List 长度为 1 时：`pad < Hk & pad < Wk`</li><li>List 长度为 2 时：`pad[0] < Hk`，`pad[1] < Wk`</li></ul> |

### pad_h

| 字段 | 说明 |
|------|------|
| 是否必填 | 非必填 |
| 数据类型 | int |
| 参数解释 | 高和宽轴开始和结束的 Padding，`pad` 和 `pad_h`/`pad_w` 不能同时提供，默认值为 0 |
| 规格限制 | `pad_h < Hk` |

### pad_w

| 字段 | 说明 |
|------|------|
| 是否必填 | 非必填 |
| 数据类型 | int |
| 参数解释 | 高和宽轴开始和结束的 Padding，`pad` 和 `pad_h`/`pad_w` 不能同时提供，默认值为 0 |
| 规格限制 | `pad_w < Wk` |

### stride

| 字段 | 说明 |
|------|------|
| 是否必填 | 非必填 |
| 数据类型 | ListInt |
| 参数解释 | 高和宽轴的 stride。`stride` 和 `stride_h`/`stride_w` 不能同时提供，默认值为 1；`stride` 的 List 长度最大为 2 |
| 规格限制 | <ul><li>List 长度为 1 时：`0 <= stride * stride < 256`</li><li>List 长度为 2 时：`1 <= stride[0] <= 63`，`1 <= stride[1] <= 63`，`1 <= stride[0] * stride[1] < 256`</li></ul> |

### stride_h

| 字段 | 说明 |
|------|------|
| 是否必填 | 非必填 |
| 数据类型 | int |
| 参数解释 | 高和宽轴的 stride。`stride` 和 `stride_h`/`stride_w` 不能同时提供，默认值为 1 |
| 规格限制 | `1 <= stride <= 63` |

### stride_w

| 字段 | 说明 |
|------|------|
| 是否必填 | 非必填 |
| 数据类型 | int |
| 参数解释 | 高和宽轴的 stride。`stride` 和 `stride_h`/`stride_w` 不能同时提供，默认值为 1 |
| 规格限制 | `1 <= stride <= 63` |

### dilation

| 字段 | 说明 |
|------|------|
| 是否必填 | 非必填 |
| 数据类型 | ListInt |
| 参数解释 | filter 的高和宽轴的放大系数，List 长度最大为 2，默认值为 1 |
| 规格限制 | 支持 `1~255`，配置后 `(kernel - 1) * dilation + 1 < 256` |

### group

| 字段 | 说明 |
|------|------|
| 是否必填 | 非必填 |
| 数据类型 | int |
| 参数解释 | 无 |
| 规格限制 | `group` 能被 channel 整除 |

---

## 输出

### y

| 字段 | 说明 |
|------|------|
| 是否必填 | 必填 |
| 数据类型 | float16 |
| 参数解释 | 无 |
| 规格限制 | 无 |
