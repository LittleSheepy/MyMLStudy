import cv2
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
# Create a simple model
model = torch.nn.Linear(2, 2)




# Function to change the color value
def draw_line(val):
    # Get the current positions of the trackbar
    _lambda = cv2.getTrackbarPos('lambda', 'Line Graph')
    _lambda = _lambda * 0.01
    lr_lambda = lambda epoch: _lambda ** epoch

    # Create a black image
    img = np.zeros((400,400,3), np.uint8)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    # Create a PolynomialLR scheduler
    scheduler = LambdaLR(optimizer, lr_lambda)
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
        cv2.line(img, (i-1, int(400-100*lrs[i-1])), (i, int(400-100*lrs[i])), (0,255,0), 2)

    # Show the image
    cv2.imshow('Line Graph', img)

# Create a window
cv2.namedWindow('Line Graph')

# Create a trackbar for color
cv2.createTrackbar('lambda', 'Line Graph', 0, 100, draw_line)
draw_line(1)
# Wait for any key to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()