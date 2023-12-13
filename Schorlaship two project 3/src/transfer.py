import torch
import torchvision
import torchvision.models as models
import torch.nn as nn

def get_model_transfer_learning(model_name="vgg16", n_classes=50):
    # Get the requested architecture
    if hasattr(models, model_name):
        model_transfer = getattr(models, model_name)(pretrained=True)
    else:
        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])
        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")

    # Freeze all parameters in the model
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Create a new custom classifier
    num_ftrs = model_transfer.classifier[6].in_features
    classifier = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, n_classes)
    )

    # Assign the custom classifier to the model
    model_transfer.classifier[6] = classifier

    return model_transfer

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):

    model = get_model_transfer_learning(n_classes=23)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
