import os
import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

def gen_golden_data_simple():
    input_shape = [1, 256, 6, 62, 62]
    weight_shape = [256, 256, 4, 4, 4]

    input = torch.zeros(input_shape, dtype=torch.float32) 
    weight = torch.zeros(weight_shape, dtype=torch.float32) 

    golden = torch.nn.functional.conv_transpose3d(input, weight, bias=None, stride=2, padding=3, output_padding=0, groups=1, dilation=1) 

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