import torch
from sklearn.metrics import roc_auc_score # For multi-label AUC-ROC
import numpy as np

def train_model(num_epochs,model,train_loader,val_loader,optimizer,criterion,print_every_n_batches,device):
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.float().to(device) # Ensure labels are float for BCEWithLogitsLoss

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % print_every_n_batches == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Training Loss: {avg_train_loss:.4f}")

        # Validation Loop
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad(): # Disable gradient calculation during validation
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Calculate AUC-ROC for each class and then average (macro)
        # The number of classes for ChestMNIST is 14 (info['n_classes'])
        roc_auc_macro = roc_auc_score(all_labels, all_preds, average='macro')

        print(f"Validation Loss: {avg_val_loss:.4f}, AUC-ROC (macro): {roc_auc_macro:.4f}")
        print("-" * 50)

    print("Training complete on ChestMNIST.")
