import os
import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

def gen_golden_data_simple():
    input_shape = [2, 3, 4]

    input = torch.ones(input_shape, dtype=torch.float32)

    golden = torch.sum(input, dim=1)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    
    input_f32 = input.to(torch.float32)
    output_f32 = golden.to(torch.float32)

    input_f32.detach().numpy().tofile("./input/input.bin")
    output_f32.detach().numpy().tofile("./output/output.bin")
    print(output_f32)


if __name__ == "__main__":
    gen_golden_data_simple()