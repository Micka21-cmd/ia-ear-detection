import os
import time
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch_directml
from tqdm import tqdm
import matplotlib.pyplot as plt

# Initialisation du device DirectML pour exploiter la carte AMD RX580
#dml = torch_directml.device()
#device = dml  # Utiliser DirectML partout
device = torch.device("cpu")

class EarDetectionDataset(Dataset):
    def __init__(self, root_dir, transforms=None, threshold=128):
        """
        root_dir : dossier contenant les sous-dossiers "input" et "target"
        transforms : transformation appliquée sur l'image (ex. to_tensor)
        threshold : seuil pour détecter les pixels blancs dans le masque
        On ne garde que les images pour lesquelles le masque existe.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.threshold = threshold
        self.image_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")
        self.image_files = []
        
        # Parcourir les fichiers dans le dossier input et ne garder que ceux ayant un masque correspondant
        for f in os.listdir(self.image_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                target_path = os.path.join(self.target_dir, f)
                if os.path.exists(target_path):
                    self.image_files.append(f)
        if len(self.image_files) == 0:
            raise ValueError("Aucune image avec masque trouvé dans le dataset.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Charger l'image d'entrée
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Impossible de lire l'image {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Charger l'image masque depuis target
        target_path = os.path.join(self.target_dir, img_name)
        mask_img = Image.open(target_path).convert("L")  # conversion en niveaux de gris
        mask_arr = np.array(mask_img)
        
        # Calculer la boîte englobante à partir du masque
        # On considère les pixels avec une intensité > threshold comme faisant partie de l'oreille.
        white_pixels = np.where(mask_arr > self.threshold)
        if white_pixels[0].size == 0 or white_pixels[1].size == 0:
            raise ValueError(f"Aucune région détectée dans le masque pour l'image {img_name}")
        
        y_min = np.min(white_pixels[0])
        y_max = np.max(white_pixels[0])
        x_min = np.min(white_pixels[1])
        x_max = np.max(white_pixels[1])
        
        # Vérifier que la boîte a une largeur et une hauteur positives
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"Boîte invalide calculée pour l'image {img_name}")
        
        boxes = [[float(x_min), float(y_min), float(x_max), float(y_max)]]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # 1 = oreille
        
        target = {"boxes": boxes, "labels": labels}
        
        if self.transforms:
            image = self.transforms(image)
        return image, target

def detection_transform(image):
    return F.to_tensor(image)

def collate_fn(batch):
    return tuple(zip(*batch))

def train_detection_model(model, optimizer, scheduler, data_loader, num_epochs, checkpoint_dir="checkpoints_detection"):
    since = time.time()
    history = {"epoch_loss": []}
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        pbar = tqdm(data_loader, desc="Training", leave=False)
        for images, targets in pbar:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
            num_batches += 1
            pbar.set_postfix(loss=losses.item())
        epoch_loss = running_loss / num_batches
        print(f"Epoch {epoch} - Average Loss: {epoch_loss:.4f}")
        history["epoch_loss"].append(epoch_loss)
        scheduler.step()
        
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint sauvegardé: {checkpoint_path}")
    
    time_elapsed = time.time() - since
    print(f"\nEntraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    return model, history

def plot_detection_history(history, output_dir="results_detection"):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["epoch_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, history["epoch_loss"], 'bo-', label="Training Loss")
    plt.title("Évolution de la Loss (modèle de détection)")
    plt.xlabel("Époque")
    plt.ylabel("Loss")
    plt.legend()
    loss_fig = os.path.join(output_dir, "detection_loss.png")
    plt.savefig(loss_fig)
    print(f"Figure de la loss sauvegardée: {loss_fig}")

def main():
    dataset_dir = "Dataset2"  # chemin racine vers Dataset2
    dataset2 = EarDetectionDataset(root_dir=dataset_dir, transforms=detection_transform)
    
    # Création du DataLoader (ajustez num_workers si nécessaire)
    data_loader = DataLoader(dataset2, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Chargement du modèle Faster R-CNN pré-entraîné et adaptation de la tête pour 2 classes (fond et oreille)
    model_det = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 0: fond, 1: oreille
    in_features = model_det.roi_heads.box_predictor.cls_score.in_features
    model_det.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model_det.to(device)

    params = [p for p in model_det.parameters() if p.requires_grad]
    optimizer_det = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler_det = torch.optim.lr_scheduler.StepLR(optimizer_det, step_size=3, gamma=0.1)

    num_epochs = 10
    model_det, history = train_detection_model(model_det, optimizer_det, scheduler_det,
                                               data_loader, num_epochs, checkpoint_dir="checkpoints_detection")
    torch.save(model_det.state_dict(), "ear_detection_model.pth")
    print("Modèle final de détection sauvegardé: ear_detection_model.pth")

    # Évaluation simple : inférence sur un batch du dataset
    model_det.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model_det(images)
            print("Sorties (boxes, labels, scores):", outputs)
            break

    plot_detection_history(history, output_dir="results_detection")

if __name__ == '__main__':
    main()
