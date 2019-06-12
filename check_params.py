from unet_parts import double_conv, double_dsconv
from unet_model import UNet, UNet_dsc
from torchsummary import summary
import torch 

model = UNet(3, 2)
model_dsc = UNet_dsc(3, 2)

# summary(model.cuda(), (3, 512, 512))
# summary(model_dsc.cuda(), (3, 512, 512))
# conv = double_conv(64, 128)
# conv_dsc = double_dsconv(64, 128)

# summary(conv.cuda(), (64,512,512))
# summary(conv_dsc.cuda(), (64, 512, 512))

torch.save(model.state_dict(), 'unet.pth')
torch.save(model_dsc.state_dict(), 'unet_dsc.pth')