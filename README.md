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
All datasets in the paper are open-source.
In folder frequency_guided_diffusion, there are the py files which are needed for the training and testing for frequency_diffusion model.
In folder networks, there are some comparison methods.
In folder dataloaders, there is the

