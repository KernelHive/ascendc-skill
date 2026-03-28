import os
import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

def gen_golden_data_simple():
    input_shape = [2, 2, 7, 7, 7]
    weight_shape = [2, 2, 1, 1, 1]

    input = torch.ones(input_shape, dtype=torch.float32) 
    weight = torch.ones(weight_shape, dtype=torch.float32) 

    weight.requires_grad_(True)
    strides = [1, 1, 1]
    pads = [0, 0, 0]
    dilations = [1, 1, 1]

    golden = F.conv3d(input, weight, stride=strides, padding=pads, dilation=dilations)

    gradOutput = torch.ones_like(golden)
    golden.backward(gradOutput)
    output = weight.grad

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    
    input_f32 = input.to(torch.float32)
    weight_f32 = weight.to(torch.float32)
    gradOutput_f32 = gradOutput.to(torch.float32)
    output_f32 = output.to(torch.float32)

    input_bf16 = input_f32.detach().numpy().astype(tf.bfloat16.as_numpy_dtype)
    weight_bf16 = weight_f32.detach().numpy().astype(tf.bfloat16.as_numpy_dtype)
    gradOutput_bf16 = gradOutput_f32.detach().numpy().astype(tf.bfloat16.as_numpy_dtype)
    output_bf16 = output_f32.detach().numpy().astype(tf.bfloat16.as_numpy_dtype)

    input_f32.detach().numpy().tofile("./input/input.bin")
    weight_f32.detach().numpy().tofile("./input/weight.bin")
    gradOutput_f32.detach().numpy().tofile("./input/gradOutput.bin")
    output_f32.detach().numpy().tofile("./output/output.bin")
    print(output_f32)


if __name__ == "__main__":
    gen_golden_data_simple()