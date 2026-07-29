import torch
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)

# Попытка вызвать cuBLAS
try:
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    c = torch.mm(a, b)
    print("✅ cuBLAS работает!")
except Exception as e:
    print("❌ Ошибка:", e)
    
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available()) 
print('CUDA version:', torch.version.cuda)
print('GPU device:', torch.cuda.get_device_name(0))
print('GPU count:', torch.cuda.device_count())