import sys
import torch
print('script-executable:', sys.executable)
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
print('cuda_device_count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(i)
    print(i, prop.name, prop.total_memory)
