import torch
import torch.nn as nn

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, padding=1)(x)
    mu_y = nn.AvgPool2d(3, 1, padding=1)(y)

    sigma_x = nn.AvgPool2d(3, 1, padding=1)(x**2) - mu_x**2
    sigma_y = nn.AvgPool2d(3, 1, padding=1)(y**2) - mu_y**2
    sigma_xy = nn.AvgPool2d(3, 1, padding=1)(x * y) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d
    return SSIM

