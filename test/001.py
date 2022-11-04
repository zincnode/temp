from cProfile import label
from logging import critical
from os import pread
from re import T
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet50().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)
labels = torch.randn(5, 1000).to(device)
model.train()

def train():
    output = model(inputs)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory = True) as prof:
    with record_function("model_inference"):
        train()

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="cuda_time_total"))

prof.export_chrome_trace("trace.json")
# prof.export_stacks("./profiler_stacks.txt", "self_cuda_time_total")