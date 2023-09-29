import torch
import torch.nn as nn
import torchvision
import sklearn
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import random
from voc_dataset import VOCDataset
from collections import defaultdict
import utils

def hex_to_rgb(hex):
        rgb = []
        for i in (0, 2, 4):
            decimal = int(hex[i:i+2], 16)
            rgb.append(decimal)
  
        return np.asarray(tuple(rgb))
def rgb_to_hex(r, g, b):
    return '{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    batch = 1000
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

    feature_extract = nn.Sequential(*list(model.children())[:-1])

    test_loader = utils.get_data_loader(
                'voc', train=False, batch_size=batch, split='test', inp_size=224)

    feature_extract.eval()

    data, target, wgt = next(iter(test_loader))

    output=feature_extract(data).squeeze().detach().numpy()
    target = target.squeeze().detach().numpy()
    wgt = wgt.squeeze().detach().numpy()
    target = target*wgt
    output_embedded = TSNE(n_components=2, learning_rate='auto',
                init='random', perplexity=3).fit_transform(output)

    max_classes = int(np.max(np.sum(target,axis = 1)))
    print(max_classes)
    labelList = np.zeros((batch,max_classes))
    emptyList = []
    for i in range(len(target)):
        if np.sum(target[i]) == 0:
            emptyList.append(i)
            print("continuing")
            continue
        num_classes = int(np.sum(target[i]))
        labelList[i,:num_classes] = (np.where(target[i,:] == 1)[0])
    labelList = np.delete(labelList,emptyList,axis = 0)
    output_embedded = np.delete(output_embedded, emptyList,axis = 0)
    # print(output_embedded.shape)
    # print(labelList.shape)
    colorMap = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

    class_colors = {}
    stringDict = {}
    for i, image_labels in enumerate(labelList):
        image_colors = []
        if len(np.where(image_labels!=0)[0])==1  or sum(image_labels) == 0:
            label = image_labels[0]
            class_color = colorMap[int(label)]
            class_colors[str(image_labels)]= class_color
            stringDict[str(image_labels)] = image_labels
        else:
            mean_list = []
            num_classes = int(np.sum(image_labels))
            for label in image_labels[:num_classes]:
                class_color = colorMap[int(label)][1:]

                color = hex_to_rgb(class_color)
 
                mean_list.append(np.array(color))

            mean_color = np.sum(np.array(mean_list),axis = 0)/len(mean_list)
            mean_color = '#'+rgb_to_hex(mean_color[0],mean_color[1],mean_color[2])

            class_colors[str(image_labels)] = mean_color
            stringDict[str(image_labels)] = image_labels
        


    legend_labels = list(class_colors.keys())
    legend_colors = [class_colors[key] for key in legend_labels]

    plt.figure(figsize=(12, 10))
    for i, label in enumerate(legend_labels):

        actualLabel = stringDict[label]
        if (len(np.where(actualLabel!=0)[0])==1 and actualLabel[0]!=0) or np.sum(actualLabel) == 0:
            label = int(actualLabel[0])
        else:
            num_zeros = len(np.where(actualLabel == 0)[0])
            if actualLabel[0] == 0:
                label = actualLabel[:int(len(actualLabel)-num_zeros+1)]
            else:
                label = actualLabel[:int(len(actualLabel)-num_zeros)]
            ##print(label)
        x_in_y = np.equal(np.array(labelList), actualLabel).all(axis=1)

        points = output_embedded[np.where(x_in_y)[0]]
        plt.scatter(points[:, 0], points[:, 1], c=legend_colors[i], label=label)

    # Add legend
    plt.legend(loc='best',ncol = 2, fontsize="5", bbox_to_anchor=(1, 1))

    # Show the plot
    plt.title('t-SNE Projection of ImageNet Features')
    plt.show()



