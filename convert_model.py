import importlib.util
import sys
from pathlib import Path

input_dir = Path(__file__).parent / 'input'

## BirdNET+ V3.0 preview3
backbone = 'tf_efficientnetv2_s_in21k'
nClasses = 11560 # 1225

## Load model definition from model.py
model_module_path = input_dir / 'model.py'
module_name = 'model_module'
spec = importlib.util.spec_from_file_location(module_name, model_module_path)
model_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = model_module
spec.loader.exec_module(model_module)


model = model_module.MultiHead01(backbone, nClasses, pretrained=False)

model.eval() # ?