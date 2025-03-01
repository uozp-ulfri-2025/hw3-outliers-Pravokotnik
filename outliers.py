from sklearn.metrics import euclidean_distances
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision.models import squeezenet1_1
import os

# Load the pre-trained SqueezeNet 1.1 model
model = squeezenet1_1(pretrained=True)
model.eval()

# Define the image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def embed(fn):
    """ Embed the given image with SqueezeNet 1.1.

    Consult https://pytorch.org/hub/pytorch_vision_squeezenet/

    The above link also uses softmax as the final transformation;
    avoid that final step. Convert the output tensor into a numpy
    array and return it.
    """

    img = Image.open(fn)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        output = model(batch_t)
    return output.squeeze().numpy()

def read_data(path):
    """
    Preberi vse slike v podani poti, jih pretvori v "embedding" in vrni rezultat
    kot slovar, kjer so ključi imena slik, vrednosti pa pripadajoči vektorji.
    """

    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
    data = {}

    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, start=path)
                rel_path = rel_path.replace(os.sep, '/')
                data[rel_path] = embed(full_path)
    return data

def euclidean_dist(r1, r2):
    return np.linalg.norm(r1 - r2)

def cosine_dist(d1, d2):
    """
    Vrni razdaljo med vektorjema d1 in d2, ki je smiselno
    pridobljena iz kosinusne podobnosti.
    """

    dot_product = np.dot(d1, d2)
    norm_d1 = np.linalg.norm(d1)
    norm_d2 = np.linalg.norm(d2)
    if norm_d1 == 0 or norm_d2 == 0:
        return 1.0
    cosine_sim = dot_product / (norm_d1 * norm_d2)
    return 1.0 - cosine_sim

def silhouette(el, clusters, data, distance_fn=euclidean_dist):
    """
    Za element el ob podanih podatkih data (slovar vektorjev) in skupinah
    (seznam seznamov nizov: ključev v slovarju data) vrni silhueto za element el.
    """

    current_cluster = next((cluster for cluster in clusters if el in cluster), None)

    if current_cluster is None:
        return 0.0
    if len(current_cluster) == 1:
        return 0.0
    
    same_elements = [x for x in current_cluster if x != el]
    a = 0.0
    if same_elements:
        a = np.mean([distance_fn(data[el], data[x]) for x in same_elements])
    
    b = np.inf
    for cluster in clusters:
        if cluster == current_cluster:
            continue
        avg = np.mean([distance_fn(data[el], data[x]) for x in cluster])
        if avg < b:
            b = avg
    if b == np.inf:
        return 0.0

    if max(a, b) == 0:
        return 0.0
    return (b - a) / max(a, b)

def silhouette_average(data, clusters, distance_fn=euclidean_dist):
    """
    Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrni povprečno silhueto.
    """

    total = 0.0
    count = 0

    for cluster in clusters:
        for el in cluster:
            s = silhouette(el, clusters, data, distance_fn)
            total += s
            count += 1
    return total / count if count else 0.0

def group_by_dir(names):
    """ Generiraj skupine iz direktorijev, v katerih se nahajajo slike """

    groups = {}
    for name in names:
        if '/' in name:
            dir_part = name.rsplit('/', 1)[0]
        else:
            dir_part = ''
        groups.setdefault(dir_part, []).append(name)
    return list(groups.values())

def order_by_decreasing_silhouette(data, clusters):
    elements = [el for cluster in clusters for el in cluster]
    scores = {el: silhouette(el, clusters, data, cosine_dist) for el in elements}
    return sorted(elements, key=lambda x: scores[x], reverse=True)

if __name__ == "__main__":
    data = read_data("traffic-signs")
    clusters = group_by_dir(data.keys())
    ordered = order_by_decreasing_silhouette(data, clusters)
    print("ATYPICAL TRAFFIC SIGNS")
    for o in ordered[-3:]:
        print(o)
