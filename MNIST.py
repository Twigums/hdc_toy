# from torchhd examples, just made it more clear for myself

import torch
import torch.nn as nn

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid

from tqdm import tqdm

from torchmetrics import Accuracy

from load_data import load_MNIST

DIMENSIONS = 10000
IMAGE_SIZE = 28
NUM_LEVELS = 1000
BATCH_SIZE = 1

class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()

        self.flatten = torch.nn.Flatten()

        # first param should always be larger than len(vecor) being encoded
        # second should be number of dimensions to encode to
        self.position = embeddings.Random(size * size, out_features)

        # encodes the linspace of the number of levels as hdv
        self.value = embeddings.Level(levels, out_features)

    def forward(self, x):
        x = self.flatten(x)

        # bind the learnable weights from random hdv with encoded levels of x
        x_bind = torchhd.bind(self.position.weight, self.value(x))

        # creates multiset with dim x_bind and size 0
        x_multiset = torchhd.multiset(x_bind)

        # make x binary again by hard quantize
        x_hquantize = torchhd.hard_quantize(x_multiset)

        return x_hquantize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")

train_dataset, test_dataset, train_loader, test_loader = load_MNIST(BATCH_SIZE, device)
encoder = Encoder(DIMENSIONS, IMAGE_SIZE, NUM_LEVELS)
encoder = encoder.to(device)

num_classes = len(train_dataset.classes)
model = Centroid(DIMENSIONS, num_classes)
model = model.to(device)

with torch.no_grad():
    for images, labels in tqdm(train_loader, desc = "Training"):
        images = images.to(device)
        labels = labels.to(device)

        images_encode = encoder(images)
        model.add(images_encode, labels)

accuracy = Accuracy("multiclass", num_classes = num_classes)

with torch.no_grad():

    # make all prototype vectors into unit vectors
    model.normalize()

    for images, labels in tqdm(test_loader, desc = "Testing"):
        images = images.to(device)

        images_encode = encoder(images)

        # `dot = True` makes inferences efficient via normalize()
        pred = model(images_encode, dot = True)
        accuracy.update(pred.cpu(), labels)

print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
