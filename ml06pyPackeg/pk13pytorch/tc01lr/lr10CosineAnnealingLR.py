import cv2
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def draw_line(val):
    pass

# 使用余弦退火设置每个参数组的学习率
if __name__ == '__main__':
    model = torch.nn.Linear(2, 2)
    cv2.namedWindow('Line Graph')
    cv2.createTrackbar('eta_min', 'Line Graph', 10, 100, draw_line)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # Create a black image
        img = np.zeros((400, 400, 3), np.uint8)
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        # Get the current positions of the trackbar
        eta_min = cv2.getTrackbarPos('eta_min', 'Line Graph')
        eta_min = eta_min * 0.01
        # Create a PolynomialLR scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=400, eta_min=eta_min)
        # Store learning rates for each epoch
        lrs = []

        # Simulate 100 epochs
        for epoch in range(400):
            # Update optimizer
            optimizer.step()
            # Update learning rate
            scheduler.step()
            # Save current learning rate
            lrs.append(optimizer.param_groups[0]['lr'])

        # Draw the line graph
        for i in range(1, len(lrs)):
            cv2.line(img, (i - 1, int(400 - 300 * lrs[i - 1])), (i, int(400 - 300 * lrs[i])), (0, 255, 0), 2)

        cv2.line(img, (0, 400), (400, 400), (255, 255, 0), 2)

        # Show the image
        cv2.imshow('Line Graph', img)

    cv2.destroyAllWindows()
