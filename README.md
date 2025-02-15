# Implement-for-CPDNet
This is source code for CPDNet
## Installation

requirements:

- Python >= 3.8
- PyTorch >= 1.10.0
- torchvision >= 0.11.1

Install other requirements by:
```
pip install -r requirements.txt
```
Conducet the operation of Crop-Paste by:
```
python Example of crop-paste  #for LA training
```
To train a model：
```
python Defect_semi_train.py  #for semi-supervised training
python diffusion_train.py  #for diffusion training
``` 

To test a model：
```
python evaluation for semi.py  #for semi-supervised testing
python evaluation for diffusion.py  #for diffusion testing
```
