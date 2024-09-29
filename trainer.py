import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast

# Flag for mixed precision training
MixedPrecession = True


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 path2write: str = None,
                 scaler: object = None,
                 ):

        # Initialize parameters
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.scaler = scaler
        self.epoch = epoch
        self.notebook = notebook
        self.training_loss = []  # To store training losses
        self.validation_loss = []  # To store validation losses
        self.learning_rate = []  # To store learning rates
        self.path2write = path2write

        # Set up TensorBoard logging
        LOG_DIR = path2write + 'Log/'
        self.writer_train = SummaryWriter(os.path.join(LOG_DIR, "train"))
        self.writer_val = SummaryWriter(os.path.join(LOG_DIR, "val"))

    def run_trainer(self):
        from tqdm import tqdm, trange

        self.model.to(self.device)  # Move model to device (GPU or CPU)
        progressbar = trange(self.epochs, desc='Progress', disable=False)
        loss_max = 1000  # Initialize maximum loss for saving best model

        for i in progressbar:
            self.lr_scheduler.step(i)  # Step the learning rate scheduler
            self.epoch += 1  # Increment epoch counter

            # Training phase
            train_loss, train_accu = self._train()
            self.writer_train.add_scalar("Loss", train_loss, i)
            self.writer_train.add_scalar("Accuracy", train_accu, i)

            # Validation phase
            val_loss, val_accu = self._validate()
            self.writer_val.add_scalar("Loss", val_loss, i)
            self.writer_val.add_scalar("Accuracy", val_accu, i)
            self.writer_train.add_scalar("Learning_rate", self.optimizer.param_groups[0]['lr'], i)

            # Save model if validation loss improves
            if loss_max > val_loss:
                print('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(loss_max, val_loss))
                save_model = "bestmodel" + str(i) + ".pth"
                torch.save(self.model.state_dict(), os.path.join(self.path2write, save_model))
                loss_max = val_loss  # Update max loss

        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):
        from tqdm import tqdm

        self.model.train()  # Set model to training mode
        train_losses = []  # To accumulate training losses
        correct = 0
        total = 0

        # Batch iterator for training
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          disable=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # Move data to device
            self.optimizer.zero_grad()  # Clear gradients

            if MixedPrecession:
                with autocast():  # Mixed precision context
                    out = self.model(input)  # Forward pass
                    # for Segformer scale the logits to the size of the label
                    # out = torch.nn.functional.interpolate(
                    #     out.logits,
                    #     size=(512, 512),
                    #     mode="nearest")
                    loss = self.criterion(out, torch.argmax(target, axis=1))  # Compute loss
                self.scaler.scale(loss).backward()  # Scale loss and perform backward pass
                self.scaler.step(self.optimizer)  # Step the optimizer
                self.scaler.update()  # Update the scaler
            else:
                out = self.model(input)  # Forward pass
                # for Segformer scale the logits to the size of the label
                # out = torch.nn.functional.interpolate(
                #     out.logits,
                #     size=(512, 512),
                #     mode="nearest")
                loss = self.criterion(out, torch.argmax(target, axis=1))  # Compute loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update parameters

            train_losses.append(loss.detach().cpu())  # Store the loss

            # Calculate accuracy
            _, predicted = out.max(1)
            _, label = target.max(1)
            total += (target.size(0) * target.size(2) * target.size(3))
            correct += predicted.eq(label).sum().item()
            batch_iter.set_description(f'Training: (loss {loss:.4f})')  # Update progress bar

        self.training_loss.append(np.mean(train_losses))  # Store average training loss
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])  # Store current learning rate

        batch_iter.close()
        accu = correct / total  # Calculate accuracy

        return np.mean(train_losses), accu

    def _validate(self):
        from tqdm import tqdm

        self.model.eval()  # Set model to evaluation mode
        valid_losses = []  # To accumulate validation losses
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          disable=False)

        total = 0
        correct = 0
        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # Move data to device

            with torch.no_grad():  # Disable gradient calculation
                if MixedPrecession:
                    with autocast():  # Mixed precision context
                        out = self.model(input)  # Forward pass
                        # for Segformer scale the logits to the size of the label
                        # out = torch.nn.functional.interpolate(
                        #     out.logits,
                        #     size=(512, 512),
                        #     mode="nearest")
                        loss = self.criterion(out, torch.argmax(target, axis=1))  # Compute loss
                else:
                    out = self.model(input)  # Forward pass
                    # out = torch.nn.functional.interpolate(
                    #     out.logits,
                    #     size=(512, 512),
                    #     mode="nearest")
                    loss = self.criterion(out, torch.argmax(target, axis=1))  # Compute loss

            valid_losses.append(loss.detach().cpu())  # Store validation loss

            # Calculate accuracy
            _, predicted = out.max(1)
            _, label = target.max(1)
            total += (target.size(0) * target.size(2) * target.size(3))
            correct += predicted.eq(label).sum().item()
            batch_iter.set_description(f'Validation: (loss {loss:.4f})')  # Update progress bar

        self.validation_loss.append(np.mean(valid_losses))  # Store average validation loss

        batch_iter.close()
        accu = correct / total  # Calculate accuracy
        return np.mean(valid_losses), accu  # Return mean validation loss and accuracy

