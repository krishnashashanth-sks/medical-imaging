from tqdm.auto import tqdm

def train_model(num_epochs,model,train_dataloader,optimizer,criterion,device):
    # --- Training Loop ---
    print("\nStarting training...")
    for epoch in tqdm(range(num_epochs)):
        model.train() # Set model to training mode
        running_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_dataloader):
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

        # Handle cases where `running_loss` might not be zero at end of epoch if not divisible by 100
        if (len(train_dataloader) % 100 != 0 and running_loss > 0) or (running_loss == 0 and len(train_dataloader) == 0):
            avg_loss = running_loss / (len(train_dataloader) % 100 if len(train_dataloader) % 100 != 0 else 1) # Avoid division by zero
        else:
            avg_loss = 0.0 # If all batches were printed already or dataloader is empty

        print(f'Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {avg_loss:.4f}')

    print("Training complete!")
