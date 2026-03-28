import os
import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

def gen_golden_data_simple():
    input_shape = [4, 320, 80, 80]
    weight_shape = [320, 320, 3, 3]

    input = torch.zeros(input_shape, dtype=torch.float32) 
    weight = torch.zeros(weight_shape, dtype=torch.float32) 

    golden = torch.nn.functional.conv_transpose2d(input, weight, bias=None, stride=1, padding=1, output_padding=0, groups=1, dilation=1) 

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    
    input_f32 = input.to(torch.float32)
    weight_f32 = weight.to(torch.float32)
    golden_f32 = golden.to(torch.float32)


    input_f32.detach().numpy().tofile("./input/input.bin")
    weight_f32.detach().numpy().tofile("./input/weight.bin")
    golden_f32.detach().numpy().tofile("./output/output.bin")
    print(golden_f32)


if __name__ == "__main__":
    gen_golden_data_simple()