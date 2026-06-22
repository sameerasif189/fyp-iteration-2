import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

import torch
from src.iteration2.fusion_model import FusionGenerator, NOISE_DIM

model = FusionGenerator()
ckpt = torch.load('models/fusion_generator.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print("Loaded fusion model, epoch=%d, loss=%.6f" % (ckpt['epoch'], ckpt['loss']))

dummy_level = torch.tensor([3], dtype=torch.long)
dummy_noise = torch.randn(1, NOISE_DIM)
torch.onnx.export(model, (dummy_level, dummy_noise), 'models/fusion_generator.onnx',
    input_names=['stress_level', 'noise'], output_names=['parameters'],
    dynamic_axes={'stress_level': {0: 'batch'}, 'noise': {0: 'batch'}, 'parameters': {0: 'batch'}},
    opset_version=18, dynamo=False)
print("ONNX exported to models/fusion_generator.onnx")
