import torch
import torch.nn as nn
import torch.nn as nn

class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        r = self.upscale_factor
        c = channels // (r ** 2)
        x = x.view(batch_size, c, r, r, height, width)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(batch_size, c, height * r, width * r)
        return x
tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(1,-1,1,1)

pixel_shuffle = nn.PixelShuffle(2)
out = pixel_shuffle(tensor)
print(out, out.shape)

pixel_shuffle2 = PixelShuffle(2)
out2 = pixel_shuffle2(tensor)
print(out2, out2.shape)



