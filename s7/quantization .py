import torch
import time
from torchsummary import summary

torch.set_num_threads(1)

t1 = torch.rand(5)

t1_q = torch.quantize_per_tensor(t1, 0.1, 10, torch.quint8)

t1_dq = t1_q.dequantize()

print("Original tensor", t1, t1.dtype)

print("Quantize tensor", t1_q, t1_q.dtype)

print("Dequantize tensor", t1_dq, t1_dq.dtype)



print("##########################")

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)

        return x

model_fp32 = M().cpu()
# create a quantized model instance
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights

# run the model
input_fp32 = torch.randn(4, 4, 4, 4)

start32 = time.time()


for i in range(10000):
    res32 = model_fp32(input_fp32)


end32 = time.time()

start8 = time.time()

for i in range(10000):
    
    res8 = model_int8(input_fp32)


end8 = time.time()




print("Quantize model", end8-start8)
print("Normal model", end32-start32)

