import torch
import torch.nn as nn
def fft_mse_loss(img1, img2):
    img1_fft = torch.fft.fftn(img1, dim=(2,3),norm="ortho")
    img2_fft = torch.fft.fftn(img2, dim=(2,3),norm="ortho")
    # Splitting x and y into real and imaginary parts
    x_real, x_imag = torch.real(img1_fft), torch.imag(img1_fft)
    y_real, y_imag = torch.real(img2_fft), torch.imag(img2_fft)
    # Calculate the MSE between the real and imaginary parts separately
    mse_real = torch.nn.MSELoss()(x_real, y_real)
    mse_imag = torch.nn.MSELoss()(x_imag, y_imag)
    return mse_imag+mse_real

def content_loss(x, y):
    criterion = nn.L1Loss()
    criterion = criterion.cuda()
    loss = criterion(x,y)
    return loss

def image_compare_loss(x, y, alpha=0.3):
    # Calculation of MSE loss in the frequency domain
    loss_fft = fft_mse_loss(x, y)
    # Calculating multilevel discrete wavelet transform MSE losses
    loss_content = content_loss(x, y)

    return 0.7*loss_content + alpha*loss_fft