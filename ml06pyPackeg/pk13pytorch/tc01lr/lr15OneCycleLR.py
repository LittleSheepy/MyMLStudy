import cv2
import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR


def draw_line(val):
    pass


# 当度量停止改进时，降低学习率。
# 一旦学习停滞，模型通常会受益于将学习率降低2-10倍。
# 这个调度器读取一个度量量，如果在“耐心”epoch数量上没有看到任何改进，则学习率会降低。

if __name__ == '__main__':
    model = torch.nn.Linear(2, 2)
    cv2.namedWindow('Line Graph')
    cv2.createTrackbar('max_lr', 'Line Graph', 100, 1000, draw_line)
    cv2.createTrackbar('epochs', 'Line Graph', 100, 100, draw_line)
    cv2.createTrackbar('steps_per_epoch', 'Line Graph', 10, 100, draw_line)
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
        max_lr = cv2.getTrackbarPos('max_lr', 'Line Graph')
        epochs = cv2.getTrackbarPos('epochs', 'Line Graph')
        steps_per_epoch = cv2.getTrackbarPos('steps_per_epoch', 'Line Graph')
        max_lr = max_lr * 0.01
        epochs = epochs if epochs > 0 else 1
        steps_per_epoch = steps_per_epoch if steps_per_epoch > 0 else 1
        scheduler = OneCycleLR(optimizer,
                 max_lr=max_lr,
                 total_steps=None,
                 epochs=epochs,
                 steps_per_epoch=steps_per_epoch,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4,
                 three_phase=False,)
        # Store learning rates for each epoch
        lrs = []

        # Simulate 100 epochs
        epoches = epochs * steps_per_epoch
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
        cv2.resizeWindow('Line Graph', 1000, 400)
    cv2.destroyAllWindows()
