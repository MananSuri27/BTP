import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from model import DialogueModel
from graphs import DialogueDataset
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-6

    def forward(self, probs, target):
        # input are not the probabilities, they are just the CNN output vector
        # input and target shape: (batch_size, num_classes)
        one_hot_target = F.one_hot(target, num_classes=probs.shape[1]).float()

        # Compute focal loss
        focal_loss = -torch.sum(torch.pow(1 - probs + self.eps, self.gamma).mul(one_hot_target).mul(torch.log(probs + self.eps)), dim=1)

        return focal_loss.mean()

# Define your model, dataset, and dataloaders
model = DialogueModel(768, 512, 4, 7)
train_dataset = DialogueDataset("train")
val_dataset = DialogueDataset("test")
test_dataset = DialogueDataset("test")

train_dataloader = GraphDataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = GraphDataLoader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = GraphDataLoader(test_dataset, batch_size=4, shuffle=False)

# Define loss functions for each task
criterion_task1 = FocalLoss()
criterion_task2 = FocalLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=8e-4)
# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define your dataloaders and other necessary components

for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss_epoch = 0.0
    
    for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        # Move data to device
        g, trees = batch
        g = g.to(device)
        trees = trees.to(device)
        
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output1, output2 = model(g, trees, g.ndata['embeddings'])
        
        # Compute losses
        loss1 = criterion_task1(output1, g.ndata['label_da'])
        loss2 = criterion_task2(output2, g.ndata['label_er'])
        
        # Total loss (you can weight them if needed)
        total_loss = loss1 + 3*loss2
        total_loss.backward()
        optimizer.step()
        
        total_loss_epoch += total_loss.item()

        # Validation at the 50% point of the epoch
        if i == len(train_dataloader) // 2:
            model.eval()
            val_losses_task1 = 0.0
            val_losses_task2 = 0.0
            task1_predictions = []
            task2_predictions = []
            task1_labels = []
            task2_labels = []

            with torch.no_grad():
                for val_batch in tqdm(val_dataloader, desc=f"Validation (Mid-Epoch)"):
                    g, trees = val_batch
                    g = g.to(device)
                    trees = trees.to(device)
                    val_output1, val_output2 = model(g, trees, g.ndata['embeddings'])
                    
                    val_losses_task1 += criterion_task1(val_output1, g.ndata['label_da']).item()
                    val_losses_task2 += criterion_task2(val_output2, g.ndata['label_er']).item()

                    task1_predictions.extend(torch.argmax(val_output1, dim=1).cpu().numpy())
                    task2_predictions.extend(torch.argmax(val_output2, dim=1).cpu().numpy())
                    task1_labels.extend(g.ndata['label_da'].cpu().numpy())
                    task2_labels.extend(g.ndata['label_er'].cpu().numpy())

            # Calculate metrics at the 50% point of the epoch
            accuracy_task1 = accuracy_score(task1_labels, task1_predictions)
            accuracy_task2 = accuracy_score(task2_labels, task2_predictions)

            precision_task1 = precision_score(task1_labels, task1_predictions, average='macro')
            recall_task1 = recall_score(task1_labels, task1_predictions, average='macro')
            f1_task1 = f1_score(task1_labels, task1_predictions, average='macro')

            precision_task2 = precision_score(task2_labels, task2_predictions, average='macro')
            recall_task2 = recall_score(task2_labels, task2_predictions, average='macro')
            f1_task2 = f1_score(task2_labels, task2_predictions, average='macro')

            avg_val_loss_task1 = val_losses_task1 / len(val_dataloader)
            avg_val_loss_task2 = val_losses_task2 / len(val_dataloader)

            print(f'Mid-Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss (Task 1): {total_loss_epoch / (i+1):.4f}, '
                  f'Loss (Task 2): {total_loss_epoch / (i+1):.4f}, '
                  f'Val Loss (Task 1): {avg_val_loss_task1 / (i+1):.4f}, '
                  f'Val Loss (Task 2): {avg_val_loss_task2 / (i+1):.4f}, '
                  f'Val Accuracy (Task 1): {accuracy_task1:.4f}, '
                  f'Val Accuracy (Task 2): {accuracy_task2:.4f}, '
                  f'Val Precision (Task 1): {precision_task1:.4f}, '
                  f'Val Recall (Task 1): {recall_task1:.4f}, '
                  f'Val F1 Score (Task 1): {f1_task1:.4f}, '
                  f'Val Precision (Task 2): {precision_task2:.4f}, '
                  f'Val Recall (Task 2): {recall_task2:.4f}, '
                  f'Val F1 Score (Task 2): {f1_task2:.4f}')

    # Validation at the end of the epoch
    model.eval()
    val_losses_task1 = 0.0
    val_losses_task2 = 0.0
    task1_predictions = []
    task2_predictions = []
    task1_labels = []
    task2_labels = []

    with torch.no_grad():
        for val_batch in tqdm(val_dataloader, desc=f"Validation (Mid-Epoch)"):
            g, trees = val_batch
            g = g.to(device)
            trees = trees.to(device)
            val_output1, val_output2 = model(g, trees, g.ndata['embeddings'])
            
            val_losses_task1 += criterion_task1(val_output1, g.ndata['label_da']).item()
            val_losses_task2 += criterion_task2(val_output2, g.ndata['label_er']).item()

            task1_predictions.extend(torch.argmax(val_output1, dim=1).cpu().numpy())
            task2_predictions.extend(torch.argmax(val_output2, dim=1).cpu().numpy())
            task1_labels.extend(g.ndata['label_da'].cpu().numpy())
            task2_labels.extend(g.ndata['label_er'].cpu().numpy())

    # Calculate metrics at the end of the epoch
    accuracy_task1 = accuracy_score(task1_labels, task1_predictions)
    accuracy_task2 = accuracy_score(task2_labels, task2_predictions)

    precision_task1 = precision_score(task1_labels, task1_predictions, average='macro')
    recall_task1 = recall_score(task1_labels, task1_predictions, average='macro')
    f1_task1 = f1_score(task1_labels, task1_predictions, average='macro')

    precision_task2 = precision_score(task2_labels, task2_predictions, average='macro')
    recall_task2 = recall_score(task2_labels, task2_predictions, average='macro')
    f1_task2 = f1_score(task2_labels, task2_predictions, average='macro')

    avg_val_loss_task1 = val_losses_task1 / len(val_dataloader)
    avg_val_loss_task2 = val_losses_task2 / len(val_dataloader)

    print(f'End of Epoch [{epoch+1}/{num_epochs}], '
          f'Loss (Task 1): {total_loss_epoch / len(train_dataloader):.4f}, '
          f'Loss (Task 2): {total_loss_epoch / len(train_dataloader):.4f}, '
          f'Val Loss (Task 1): {avg_val_loss_task1 / (i+1):.4f}, '
          f'Val Loss (Task 2): {avg_val_loss_task2 / (i+1):.4f}, '
          f'Val Accuracy (Task 1): {accuracy_task1:.4f}, '
          f'Val Accuracy (Task 2): {accuracy_task2:.4f}, '
          f'Val Precision (Task 1): {precision_task1:.4f}, '
          f'Val Recall (Task 1): {recall_task1:.4f}, '
          f'Val F1 Score (Task 1): {f1_task1:.4f}, '
          f'Val Precision (Task 2): {precision_task2:.4f}'
          f'Val Recall (Task 2): {recall_task2:.4f}, '
          f'Val F1 Score (Task 2): {f1_task2:.4f}')


# Testing
model.eval()
test_losses_task1 = 0.0
test_losses_task2 = 0.0
task1_predictions = []
task2_predictions = []
task1_labels = []
task2_labels = []

with torch.no_grad():
    for test_batch in test_dataloader:
        g, trees = test_batch
        g = g.to(device)
        trees = trees.to(device)
        val_output1, val_output2 = model(g, trees, g.ndata['embeddings'])
        
        val_losses_task1 += criterion_task1(val_output1, g.ndata['label_da']).item()
        val_losses_task2 += criterion_task2(val_output2, g.ndata['label_er']).item()

        task1_predictions.extend(torch.argmax(val_output1, dim=1).cpu().numpy())
        task2_predictions.extend(torch.argmax(val_output2, dim=1).cpu().numpy())
        task1_labels.extend(g.ndata['label_da'].cpu().numpy())
        task2_labels.extend(g.ndata['label_er'].cpu().numpy())

# Calculate test metrics
precision_task1, recall_task1, f1_task1, _ = precision_recall_fscore_support(
    task1_labels, task1_predictions, average='macro', zero_division=0
)
precision_task2, recall_task2, f1_task2, _ = precision_recall_fscore_support(
    task2_labels, task2_predictions, average='macro', zero_division=0
)

avg_test_loss_task1 = test_losses_task1 / len(test_dataloader)
avg_test_loss_task2 = test_losses_task2 / len(test_dataloader)

print(f'Test Loss (Task 1): {avg_test_loss_task1:.4f}, '
      f'Test Loss (Task 2): {avg_test_loss_task2:.4f}, '
      f'Test Precision (Task 1): {precision_task1:.4f}, '
      f'Test Recall (Task 1): {recall_task1:.4f}, '
      f'Test F1 Score (Task 1): {f1_task1:.4f}, '
      f'Test Precision (Task 2): {precision_task2:.4f}, '
      f'Test Recall (Task 2): {recall_task2:.4f}, '
      f'Test F1 Score (Task 2): {f1_task2:.4f}')
