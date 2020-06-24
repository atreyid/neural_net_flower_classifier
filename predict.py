import argparse
import json
import PIL
import torch
import numpy as np

from math import ceil
from train import gpu_check
from torchvision import models

def load_checkpoint(checkpoint):
    checkpoint = torch.load(checkpoint)
    
    model_cls = getattr(models, checkpoint["model"])
    model = model_cls(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint["class_to_idx"]
    model.classifier = checkpoint["model_classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    new_width, new_height = (224, 224)
    pil_image = PIL.Image.open(image)
    pil_image.thumbnail((256, 256))
    
    width, height = pil_image.size
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    pil_image = pil_image.crop((left, top, right, bottom))
    np_image = np.array(pil_image)/255
    
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image - means) / std
    
    np_image = np_image.transpose(2, 0, 1)
    return np_image

def predict_img(image_path, model, device, cat_to_name, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print(device)
    model.to(device)
    model.eval()
    
    img_tensor = torch.from_numpy(process_image(image_path)).unsqueeze_(0).type(torch.FloatTensor).to(device)
    log_probs = model.forward(img_tensor)
    linear_probs = torch.exp(log_probs)
    top_probs, top_labels = linear_probs.topk(topk)
    
    # flatten tensor to array
    top_probs = torch.Tensor.cpu(top_probs).detach().numpy()[0]
    top_labels = torch.Tensor.cpu(top_labels).detach().numpy()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_labels]
    top_flowers_names = [cat_to_name[label] for label in top_labels]

    
    return top_probs, top_labels, top_flowers_names

def print_probs(probs, flowers):
    for i, j in enumerate(zip(probs, flowers)):
        print ("Rank {}:".format(i+1), "Flower name: {}, probability: {}%".format(j[1], ceil(j[0]*100)))

def main(args):
    print("Loading category names...\n")
    with open(args.cat_file, "r") as f:
        cat_to_name = json.load(f)
    
    print("Loading model from checkpoint...\n")
    model = load_checkpoint(args.checkpoint)
    
    device = gpu_check(args.gpu)
    
    print("Predicting image category...\n")
    top_probs, top_labels, top_flowers = predict_img(args.image, model, device, cat_to_name, args.top_k)
    
    print_probs(top_probs, top_flowers)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction Settings")
    parser.add_argument("image", type=str, help="Path to Image for prediction")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path")
    parser.add_argument('--top_k', default=5, type=int, help="Top K matches")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--cat_file", type=str, required=True, help="File that contains mapping for category to flower names")
    
    args = parser.parse_args()
    main(args)
