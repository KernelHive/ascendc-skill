# RoiAlignRotatedGard

## 支持的产品型号

Atlas 800I A2推理产品/

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能描述

- 本算子正向需要根据输入RoI box的参数信息，计算旋转候选框的池化结果，候选框的旋转是中心旋转。
- 由于输入为旋转候选框，因此输入box中需要包含角度特征，RoI的输入tensor形状为[N,6]，其中尾轴的6个数据代表了[batchIdx, x, y, w, h, angle]。
该算子正向功能与RoiAlign差异在于，由于候选框经过了角度信息进行旋转，因此各点的坐标需要根据旋转角度重新计算。
- 正向的过程可以视作采样求和的过程，因此反向则是通过旋转框各点坐标将梯度回传至对应的位置。

  **说明：**
  无。

## 约束与限制

- channels <= 1024
- 32 <= height <= 512
- 32 <= width <= 512
- sampling_ratio <= 4
- pooled_height <= 10
- pooled_width <= 10
- 算子输入不支持inf/nan等异常值，不支持空Tensor。
- Rois形状：必须是形状为[n, 6]的Tensor。n的取值范围为[1, 8192]。对于每个roi而言，其组成为[batch_idx, center_x, center_y, w, h, angle]，其中batch_idx取值范围在[0, N)之间，会进行强制类型- 转换（float->int），center_x、center_y、w、h为float32类型正浮点数，center_x与w的取值范围为[0, W)，center_y与h的取值范围为[0, H)，angle取值为float32类型浮点数，取值范围为[0, π)。

## 算子原型

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">RoiAlignRotated</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format和说明</td></tr>
<tr><td align="center">grad_output</td><td align="center">(N,C,H,W)</td><td align="center">float32</td><td align="center">(anchors, channels, pooled_height, pooled_width)</td></tr>
<tr><td align="center">rois</td><td align="center">(N,6)</td><td align="center">int32</td><td align="center">(anchors, 6)</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">(N,C,H,W)</td><td align="center">int32</td><td align="center">NC1HWC0/roialign后的特征图</td></tr>
</tr>
<tr><td rowspan="7" align="center">属性输入</td><td align="center">pooled_h</td><td align="center"></td><td align="center">int</td><td align="center">输出特征图的高度</td></tr>
<tr><td align="center">pooled_w</td><td align="center"></td><td align="center">int</td><td align="center">输出特征图的宽度</td></tr>
<tr><td align="center">spatial_scale</td><td align="center"></td><td align="center">float</td><td align="center">对输入框进行缩放</td></tr>
<tr><td align="center">sampling_ratio</td><td align="center"></td><td align="center">int</td><td align="center">每个输出对于输入的采样个数，默认为0</td></tr>
<tr><td align="center">aligned</td><td align="center"></td><td align="center">bool</td><td align="center">是否更好的输出对齐结果，默认True</td></tr>
<tr><td align="center">clockwise</td><td align="center"></td><td align="center">bool</td><td align="center">时针方向，默认False</td></tr>
</tr>
<tr><td align="center">grad_feature_map</td><td align="center"></td><td align="center">tensor</td><td align="center">正向输入feature_map的反向梯度,(batch_size, channels, height, width)</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">roi_align_rotated_grad</td></tr>
</table>