import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class SoftAttentionReIDExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=weights).features
        self.model.to(device).eval()
        self.transform = T.Compose([
            T.ToPILImage(), T.Resize((128, 128)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _generate_gaussian_mask(self, height, width):
        kernel_x = cv2.getGaussianKernel(width, width / 2)
        kernel_y = cv2.getGaussianKernel(height, height / 2)
        mask = np.outer(kernel_y, kernel_x)
        return mask / mask.max()

    @torch.no_grad()
    def get_embeddings(self, frame, boxes):
        crops, valid_idx = [], []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)].copy()
            if crop.size > 0 and crop.shape[0] > 10:
                h, w = crop.shape[:2]
                mask = self._generate_gaussian_mask(h, w)
                mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                masked_crop = (crop * mask_3d).astype(np.uint8)
                crops.append(self.transform(masked_crop))
                valid_idx.append(i)
        batch = torch.stack(crops).to(self.device)
        emb = torch.nn.functional.adaptive_avg_pool2d(self.model(batch), 1).flatten(1)
        return torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy(), valid_idx