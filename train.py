import argparse
import torch
from collections import OrderedDict
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import time
from os.path import isdir, join

def get_transformers():
    train = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225]
        )
    ])
    validate = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225]
        )
    ])
    test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225]
        )
    ])
    return train, validate, test

def get_data(train_dir, valid_dir, test_dir):
    train_transform, validate_transform, test_transform = get_transformers()
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    validate_data = datasets.ImageFolder(valid_dir, transform=validate_transform)
    test_data = datasets.ImageFolder(valid_dir, transform=test_transform)

    return train_data, validate_data, test_data
    
def get_loader(train_data, validate_data, test_data):
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=48, shuffle=True)
    validateloader = torch.utils.data.DataLoader(validate_data, batch_size=48)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=48)
    
    return trainloader, validateloader, testloader

def build_model(arch):
    model_cls = getattr(models, arch)
    model = model_cls(pretrained=True)
    model.name = arch

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # set classifier
    classifier = nn.Sequential(
        OrderedDict([
            ("fc1", nn.Linear(25088, 4096, bias=True)),
            ("relu", nn.ReLU()),
            ("dropout", nn.Dropout(p=0.5)),
            ("fc2", nn.Linear(4096, 102, bias=True)),
            ("output", nn.LogSoftmax(dim=1))
        ])
    )
    model.classifier = classifier
    return model

def gpu_check(gpu):
    if gpu:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return "cpu"

def validate_model(model, testloader, device):
    accuracy = 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    print(f"Accuracy achieved on test data with trained model: {accuracy/len(testloader):.3f}")
    
def save_checkpoint(model, train_data, save_dir):
    if (not save_dir or not isdir(save_dir)):
        print("No save directory provided for checkpoint. The model will not be saved!\n")
        return None
    model.class_to_idx = train_data.class_to_idx
    filename = "my_checkpoint.pth"
    fullpath = join(save_dir, filename)
    checkpoint = {
        "model": model.name,
        "model_classifier": model.classifier,
        "state_dict": model.state_dict(),
        "class_to_idx": model.class_to_idx
    }
    torch.save(checkpoint, fullpath)
    return fullpath
     
def train(model, epochs, trainloader, validateloader, device, optimizer, criterion):
    steps = 0
    running_loss = 0
    print_every = 24
    
    start = time.time()
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validateloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        batch_loss = criterion(output, labels)

                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validateloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validateloader):.3f}")
            
                running_loss = 0
                model.train()
    print(f"\nTraining finished in: {time.time() - start:.2f} seconds")
    return model

def main(args):
    train_dir = args.data_dir + "/train"
    valid_dir = args.data_dir + "/valid"
    test_dir = args.data_dir + "/test"
    
    print("Loading data...\n")
    train_data, validate_data, test_data = get_data(train_dir, valid_dir, test_dir)
    trainloader, validateloader, testloader = get_loader(train_data, validate_data, test_data)
    
    print("Building model...\n")
    model = build_model(args.arch)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    device = gpu_check(args.gpu)
    model.to(device)
    
    print("Training model...\n")
    model = train(model, args.epochs, trainloader, validateloader, device, optimizer, criterion)
    
    print("\nValidating model...\n")
    validate_model(model, testloader, device)
    
    print("\nSaving checkpoint...\n")
    saved_file_path = save_checkpoint(model, train_data, args.save_dir)
    if saved_file_path:
        print(f"Checkpoint saved in {saved_file_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Settings")
    parser.add_argument("data_dir", type=str, help="Enter a model name torchvision.models (str)")
    parser.add_argument("--arch", type=str, default="vgg13", help="Enter a model name torchvision.models (str)")
    parser.add_argument("--save_dir", type=str, help="Save directory name for checkpoints (str)")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Gradient descent learning rate (float)")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs for training (int)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training (bool)")
    args = parser.parse_args()
    main(args)
