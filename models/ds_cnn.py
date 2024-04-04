import torch


class DSCNN(torch.nn.Module):
    def __init__(self, in_channels=1, in_shape=(32, 32), ds_cnn_number=3, ds_cnn_size=64, is_classifier=False, classes_number=0):
        super(DSCNN, self).__init__()

        self.classes_number = classes_number
        self.is_classifier = is_classifier

        self.initial_convolution = self.make_features(in_channels, ds_cnn_size)
        self.dscnn_blocks = self.make_dscnn_blocks(ds_cnn_size, ds_cnn_number)
        self.pool = self.make_pool(in_shape)

        self.classifier = torch.nn.Linear(ds_cnn_size, classes_number) if self.is_classifier else None

    
    def make_features(self, in_channels, out_channels):
        layers = []
    
        layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'))
        layers.append(torch.nn.BatchNorm2d(out_channels))
        layers.append(torch.nn.ReLU(inplace=True))

        return torch.nn.Sequential(*layers)

    def make_dscnn_blocks(self, ds_cnn_size, ds_cnn_number):
        layers = []

        for i in range(ds_cnn_number):
            layers.append(torch.nn.Conv2d(in_channels=ds_cnn_size, out_channels=ds_cnn_size, kernel_size=3, groups=ds_cnn_size, padding='same'))
            layers.append(torch.nn.BatchNorm2d(ds_cnn_size))          
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Conv2d(in_channels=ds_cnn_size, out_channels=ds_cnn_size, kernel_size=1, padding='same'))
            layers.append(torch.nn.BatchNorm2d(ds_cnn_size))    
            layers.append(torch.nn.ReLU(inplace=True))

        return torch.nn.Sequential(*layers)

    def make_pool(self, in_shape):
        layers = []

        layers.append(torch.nn.AvgPool2d(in_shape))
        layers.append(torch.nn.Flatten())

        return torch.nn.Sequential(*layers)
    
    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_convolution(x)
        x = self.dscnn_blocks(x)
        x = self.pool(x)

        if self.is_classifier:
            x = self.classifier(x)

        return x
