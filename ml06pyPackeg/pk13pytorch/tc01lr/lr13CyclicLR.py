import cv2
import numpy as np
import torch
from torch.optim.lr_scheduler import CyclicLR


def draw_line(val):
    pass


# 当度量停止改进时，降低学习率。
# 一旦学习停滞，模型通常会受益于将学习率降低2-10倍。
# 这个调度器读取一个度量量，如果在“耐心”epoch数量上没有看到任何改进，则学习率会降低。

if __name__ == '__main__':
    model = torch.nn.Linear(2, 2)
    cv2.namedWindow('Line Graph')
    cv2.createTrackbar('base_lr', 'Line Graph', 50, 1000, draw_line)
    cv2.createTrackbar('max_lr', 'Line Graph', 100, 1000, draw_line)
    cv2.createTrackbar('step_size_up', 'Line Graph', 10, 1000, draw_line)
    cv2.createTrackbar('gamma', 'Line Graph', 999, 10000, draw_line)
    epoches = 1000
    High = 400
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # Create a black image
        img = np.zeros((High, epoches, 3), np.uint8)
        optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.9)
        # Get the current positions of the trackbar
        base_lr = cv2.getTrackbarPos('base_lr', 'Line Graph')
        max_lr = cv2.getTrackbarPos('max_lr', 'Line Graph')
        step_size_up = cv2.getTrackbarPos('step_size_up', 'Line Graph')
        gamma = cv2.getTrackbarPos('gamma', 'Line Graph')
        base_lr = base_lr * 0.01
        max_lr = max_lr * 0.01
        step_size_up = step_size_up + 1
        gamma = (gamma+1) * 0.001
        scheduler = CyclicLR(optimizer,
                 base_lr=base_lr,
                 max_lr=max_lr,
                 step_size_up=step_size_up,
                 step_size_down=None,
                 mode='triangular',
                 gamma=gamma,
                 scale_fn=None,
                 scale_mode='cycle',
                 cycle_momentum=True,
                 base_momentum=0.8,
                 max_momentum=0.9,)
        # Store learning rates for each epoch
        lrs = []

        # Simulate 100 epochs
        for epoch in range(epoches):
            # Update optimizer
            optimizer.step()
            # Update learning rate
            scheduler.step()
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
