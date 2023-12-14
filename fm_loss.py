import torch
from torchvision import models
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

class FeatureMatchingLoss:
    def __init__(self):
        self.model = models.vgg16(pretrained=True).features[:15]
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def __call__(self, pred_img, gt_img):
        return F.mse_loss(self.model(pred_img), self.model(gt_img))

if __name__ == "__main__":
    # Create randomly initialized predicted and true images
    pred_img = torch.rand(2, 3, 256, 256).to(device)
    true_img = torch.rand(2, 3, 256, 256).to(device)

    # Calculate FeatureMatchingLoss
    FM_loss = FeatureMatchingLoss()(pred_img, true_img)
    print(FM_loss)
