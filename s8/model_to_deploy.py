import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')

t1 = torch.rand((8, 3, 512, 512))

assert torch.allclose(model(t1)[:5], script_model(t1)[:5]), "Output not equal"
