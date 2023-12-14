import torch
from math import log10, sqrt

def calculate_mean_absolute_error(pred_img: torch.tensor, true_img: torch.tensor):
    assert pred_img.shape == true_img.shape, "Image shapes must match"
    return torch.sum(torch.abs(pred_img - true_img)).item() / true_img.numel()

def calculate_epsilon_accuracy(pred_img: torch.tensor, true_img: torch.tensor, epsilon=0.5):
    assert pred_img.shape == true_img.shape, "Image shapes must match"
    temp = torch.abs(pred_img - true_img)
    accurate_count = temp[temp < epsilon].numel()
    return accurate_count / true_img.numel()

def calculate_psnr(pred_img: torch.tensor, true_img: torch.tensor):
    assert pred_img.shape == true_img.shape, "Image shapes must match"
    psnr_sum = sum(20 * log10(255.0 / sqrt(torch.mean((p_img.float() - t_img.float()) ** 2)))
                   for p_img, t_img in zip(pred_img, true_img))
    return psnr_sum / len(pred_img)

if __name__ == "__main__":
    # Test mean absolute error
    pred_img = torch.cat([torch.ones(3, 5, 5), torch.ones(3, 5, 5) * 5]).view(2, 3, 5, 5)
    true_img = torch.cat([torch.ones(3, 5, 5) * 2, torch.ones(3, 5, 5) * 7]).view(2, 3, 5, 5)
    print(calculate_mean_absolute_error(pred_img, true_img))

    # Test epsilon accuracy
    pred_img = torch.rand(1, 3, 5, 5)
    true_img = torch.rand(1, 3, 5, 5)
    print(calculate_epsilon_accuracy(pred_img, true_img, epsilon=0.5))
    
    # Test peak signal to noise ratio
    pred_img = torch.tensor([[[[1, 2], [2, 3]]], [[[3, 4], [5, 6]]]])
    true_img = torch.tensor([[[[1, 1], [3, 2]]], [[[2, 4], [6, 6]]]])
    print(calculate_psnr(pred_img, true_img))
