# from torchhd examples, just made it more clear for myself

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchhd import embeddings, functional
from tqdm import tqdm
from load_data import load_MNIST

DIMENSIONS = 10000
IMAGE_SIZE = 28
NUM_LEVELS = 1000
BATCH_SIZE = 1
LEARNING_RATE = 1e-2
EPOCH = 2

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
        x_bind = functional.bind(self.position.weight, self.value(x))

        # creates multiset with dim x_bind and size 0
        x_multiset = functional.multiset(x_bind)

        # make x binary again by hard quantize
        x_hquantize = functional.hard_quantize(x_multiset)

        return x_hquantize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")

train_dataset, test_dataset, train_loader, test_loader = load_MNIST(BATCH_SIZE, device)
num_classes = len(train_dataset.classes)

flatten = nn.Flatten()
classify = nn.Linear(DIMENSIONS, num_classes, bias = False)
classify.to(device)
classify.weight.data.fill_(0.0)

encoder = Encoder(DIMENSIONS, IMAGE_SIZE, NUM_LEVELS)
encoder = encoder.to(device)

with torch.no_grad():

    for images, labels in tqdm(train_loader, desc = "Training"):
        images = images.to(device)
        labels = labels.to(device)

        images_encode = encoder(images)
        classify.weight[labels] += images_encode

    classify.weight[:] = F.normalize(classify.weight)

pred_correct = 0
total_data = 0

with torch.no_grad():

    for images, labels in tqdm(test_loader, desc = "Testing"):
        images = images.to(device)

        images_encode = encoder(images)
        outputs = classify(images_encode)
        pred = torch.argmax(outputs, dim = -1)
        pred_correct += (pred.cpu() == labels).sum().item()
        total_data += labels.size(0)

    print(pred_correct / total_data)
