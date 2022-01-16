import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
from ptflops import get_model_complexity_info

resnet = torchvision.models.resnet152(pretrained=True)
resnet.eval()

mobilenet = torchvision.models.mobilenet_v3_large(pretrained=True)
mobilenet.eval()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data = torchvision.datasets.CIFAR10("/tmp", train=False,
                                    download=True, transform=transform)


dataloader = DataLoader(data, batch_size=16, num_workers=16)

macs, params = get_model_complexity_info(resnet, tuple(data[0][0].shape),
                                         as_strings=False,
                                         print_per_layer_stat=False,
                                         verbose=False)

print('{:<30}  {:<8}'.format('Computational complexity ResNet: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters ResNet: ', params))

macs, params = get_model_complexity_info(mobilenet, tuple(data[0][0].shape),
                                         as_strings=False,
                                         print_per_layer_stat=False,
                                         verbose=False)

print('{:<30}  {:<8}'.format('Computational complexity Mobilnet V3 Large: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters Mobilnet V3 Large: ', params))


start = time.time()

for imgs, labels in dataloader:
    resnet(imgs)

end = time.time()

print(f"ResNet512 timing: {end - start:.2f} seconds")

start = time.time()

for imgs, labels in dataloader:
    mobilenet(imgs)

end = time.time()

print(f"Mobilnet V3 Large timing: {end - start:.2f} seconds")