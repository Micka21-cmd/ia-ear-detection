# classification_training_improved.py
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch_directml
from tqdm import tqdm
import matplotlib.pyplot as plt

# Initialisation du device DirectML
dml = torch_directml.device()
device = dml  # Utiliser DirectML partout

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, checkpoint_dir="checkpoints"):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Pour sauvegarder l'historique des métriques
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Créer le dossier de checkpoints si besoin
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print('-' * 30)

        epoch_metrics = {}

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            data_iter = tqdm(dataloaders[phase], desc=f"{phase.capitalize()}")
            for inputs, labels in data_iter:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                data_iter.set_postfix(loss=loss.item())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Sauvegarder les métriques dans epoch_metrics avec conversion sur CPU pour éviter le problème
            epoch_metrics[f"{phase}_loss"] = epoch_loss
            # Conversion du tenseur en valeur scalaire (float) pour éviter l'erreur lors du plotting
            epoch_metrics[f"{phase}_acc"] = epoch_acc.cpu().item()

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        history["train_loss"].append(epoch_metrics["train_loss"])
        history["train_acc"].append(epoch_metrics["train_acc"])
        history["val_loss"].append(epoch_metrics["validation_loss"])
        history["val_acc"].append(epoch_metrics["validation_acc"])

        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint sauvegardé: {checkpoint_path}")

    time_elapsed = time.time() - since
    print(f"\nEntraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Meilleure précision de validation: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history

def plot_history(history, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], 'bo-', label="Entraînement")
    plt.plot(epochs, history["val_loss"], 'ro-', label="Validation")
    plt.title("Evolution de la Loss")
    plt.xlabel("Époque")
    plt.ylabel("Loss")
    plt.legend()
    loss_fig = os.path.join(output_dir, "loss.png")
    plt.savefig(loss_fig)
    print(f"Figure Loss sauvegardée: {loss_fig}")

    plt.figure()
    plt.plot(epochs, history["train_acc"], 'bo-', label="Entraînement")
    plt.plot(epochs, history["val_acc"], 'ro-', label="Validation")
    plt.title("Evolution de l'Accuracy")
    plt.xlabel("Époque")
    plt.ylabel("Accuracy")
    plt.legend()
    acc_fig = os.path.join(output_dir, "accuracy.png")
    plt.savefig(acc_fig)
    print(f"Figure Accuracy sauvegardée: {acc_fig}")

def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = "Dataset1"  # chemin racine vers Dataset1
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                data_transforms[x])
                      for x in ['train', 'validation']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                   for x in ['train', 'validation']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    class_names = image_datasets['train'].classes
    print("Classes :", class_names)

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft, history = train_model(model_ft, criterion, optimizer_ft, scheduler,
                                    dataloaders, dataset_sizes, num_epochs=25, checkpoint_dir="checkpoints")
    torch.save(model_ft.state_dict(), "ear_classification_model.pth")
    print("Modèle final sauvegardé: ear_classification_model.pth")

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_size = len(test_dataset)

    model_ft.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    test_acc = running_corrects.double() / test_size
    print(f'\nPrécision sur le test : {test_acc:.4f}')

    plot_history(history, output_dir="results")

if __name__ == '__main__':
    main()
