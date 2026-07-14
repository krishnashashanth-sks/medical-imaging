from tqdm.auto import tqdm
def train_model(num_epochs,model,train_dataloader,optimizer,criterion,device):
    # --- Training Loop ---
    print("\nStarting training...")
    for epoch in tqdm(range(num_epochs)):
        model.train() # Set model to training mode
        running_loss = 0.0
        for batch_idx, (images, masks) in tqdm(enumerate(train_dataloader)):
            images = images.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0: # Print every 100 batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

        print(f'Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {running_loss / (len(train_dataloader) % 100 if len(train_dataloader) % 100 != 0 else 100):.4f}')

    print("Training complete!")

    # You might want to save the trained model
    # torch.save(model.state_dict(), 'unet_medmnist.pth')
    # print("Model saved to unet_medmnist.pth")