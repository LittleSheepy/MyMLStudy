import cv2
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def draw_line(val):
    pass


# 当度量停止改进时，降低学习率。
# 一旦学习停滞，模型通常会受益于将学习率降低2-10倍。
# 这个调度器读取一个度量量，如果在“耐心”epoch数量上没有看到任何改进，则学习率会降低。

if __name__ == '__main__':
    model = torch.nn.Linear(2, 2)
    cv2.namedWindow('Line Graph')
    cv2.createTrackbar('factor', 'Line Graph', 80, 99, draw_line)
    cv2.createTrackbar('patience', 'Line Graph', 40, 99, draw_line)
    cv2.createTrackbar('threshold', 'Line Graph', 990, 10000, draw_line)
    cv2.createTrackbar('cooldown', 'Line Graph', 0, 100, draw_line)
    cv2.createTrackbar('min_lr', 'Line Graph', 0, 100, draw_line)
    epoches = 1000
    High = 400
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # Create a black image
        img = np.zeros((High, epoches, 3), np.uint8)
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        # Get the current positions of the trackbar
        factor = cv2.getTrackbarPos('factor', 'Line Graph')
        patience = cv2.getTrackbarPos('patience', 'Line Graph')
        threshold = cv2.getTrackbarPos('threshold', 'Line Graph')
        cooldown = cv2.getTrackbarPos('cooldown', 'Line Graph')
        min_lr = cv2.getTrackbarPos('min_lr', 'Line Graph')
        factor = (factor+1) * 0.01
        patience = patience * 1
        threshold = threshold * 0.001
        min_lr = min_lr * 0.01
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience,
                 threshold=threshold, threshold_mode='rel', cooldown=cooldown,
                 min_lr=min_lr, eps=1e-8,)
        # Store learning rates for each epoch
        lrs = []

        # Simulate 100 epochs
        for epoch in range(epoches):
            # Update optimizer
            optimizer.step()
            # Update learning rate
            scheduler.step(1.0)
            # Save current learning rate
            lrs.append(optimizer.param_groups[0]['lr'])

        # Draw the line graph
        for i in range(1, len(lrs)):
            cv2.line(img, (i - 1, int(High - High * lrs[i - 1])), (i, int(High - High * lrs[i])), (0, 255, 0), 2)

        for i in range(4):
            ratio = i/4
            y_ = int(High*ratio)
            cv2.line(img, (0, y_), (epoches, y_), (255, 255, 0), 2)

        # Show the image
        cv2.imshow('Line Graph', img)

    cv2.destroyAllWindows()
