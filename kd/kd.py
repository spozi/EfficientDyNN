from abc import ABC, abstractmethod
from tqdm import tqdm
import copy



# We are going to use PyTorch to perform knowledge distillation
# I think perhaps user can input their "torch package" as knowledge distillation.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

class DistillationStrategy(ABC):
    @abstractmethod
    def distill(self, teacher_model, student_model):
        pass
    @abstractmethod
    def train(self, **kwargs):
        pass

class TraditionalDistillation(DistillationStrategy):
    def distill(self, teacher_model, student_model):
        # Implement traditional knowledge distillation
        self.teacher_model = teacher_model
        self.student_model = student_model
        pass
    
    def train(self, **kwargs):
        self.best_model_state = {}
        self.best_optimizer_state = {}
        self.best_accuracy = 0
        self.train_history = []
        """
        Train the student model based on initialized teacher_model and student_model.

        Parameters:
            **kwargs: Additional keyword arguments.
                Possible keywords:
                    - train_loader (torch.dataset): Train set
                    - val_loader (torch.dataset): Validation set
                    - epochs (int): Number of epochs
                    - learning_rate (float):  Learning rate (e.g. 0.001)
                    - T (float): Temperature controls the smoothness of the output distributions. Larger T leads to smoother distributions, thus smaller probabilities get a larger boost.
                    - soft_target_loss_weight (float): A weight assigned to the extra objective we’re about to include
                    - ce_loss_weight (float): A weight assigned to cross-entropy. Tuning these weights pushes the network towards optimizing for either objective.
                    - device (str): A specific device that will be used to run this training function
        """
        train_loader = kwargs.get('train_loader')
        if train_loader is None:
            raise ValueError("train_loader must be provided.")
        val_loader = kwargs.get('val_loader')
        if val_loader is None:
            raise ValueError("val_loader must be provided.")
        epochs = kwargs.get('epochs')
        if epochs is None:
            raise ValueError("epochs must be provided.")
        learning_rate = kwargs.get('learning_rate')
        if learning_rate is None:
            raise ValueError("learning_rate must be provided.")
        T = kwargs.get('T')
        if T is None:
            raise ValueError("T must be provided.")
        soft_target_loss_weight = kwargs.get('soft_target_loss_weight')
        if soft_target_loss_weight is None:
            raise ValueError("soft_target_loss_weight must be provided.")
        ce_loss_weight = kwargs.get('ce_loss_weight')
        if ce_loss_weight is None:
            raise ValueError("ce_loss_weight must be provided.")
        device = kwargs.get('device')
        if device is None:
            raise ValueError("device must be provided.")
        
        ce_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.student_model.parameters(), lr=learning_rate)

        self.teacher_model.eval()  # Teacher set to evaluation mode
        self.student_model.train() # Student to train mode

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                with torch.no_grad():
                    teacher_logits = self.teacher_model(inputs)

                # Forward pass with the student model
                student_logits = self.student_model(inputs)

                #Soften the student logits by applying softmax first and log() second
                soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

                # Calculate the true label loss
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = student_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()            
            train_accuracy = correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Train Accuracy: {100*train_accuracy:.2f}%")

            # Validation stage
            self.student_model.eval()  # Set model to evaluation mode
            with torch.inference_mode():
                val_correct = 0
                val_total = 0
                for val_inputs, val_labels, val_nums in tqdm(val_loader, desc=f'Validation', unit='batch'):
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = self.student_model(val_inputs)
                    _, val_predicted = val_outputs.max(1)
                    val_total += val_labels.size(0)
                    val_correct += val_predicted.eq(val_labels).sum().item()
                    
            # Calculate validation accuracy
            val_accuracy = val_correct / val_total
            print(f"Validation Accuracy: {100*val_accuracy:.2f}%")
            
            # Store the learning performance history
            self.train_history.append((running_loss, train_accuracy, val_accuracy))
            
            # Save model if validation accuracy improves
            if val_accuracy > self.best_accuracy:
                self.best_epoch = epoch
                self.best_accuracy = val_accuracy
                self.best_model_state = copy.deepcopy(self.student_model.state_dict())
                self.best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                # best_model_state = self.student_model.state_dict()
                # torch.save(best_model_state, f'{model_saved_name}.pth')
            print(f"Best accuracy ({self.best_epoch+1}): {self.best_accuracy}")
        print(f"Best validation accuracy:{self.best_accuracy} with train accuracy:{train_accuracy}")
    
    def save(self, path):
        """
            Save checkpoint of the best model.
        """
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.best_model_state.state_dict(),
            'optimizer_state_dict': self.best_optimizer_state.state_dict(),
            'train_history': self.train_history
        }, path)
        
            
class AttentionTransfer(DistillationStrategy):
    def distill(self, teacher_model, student_model):
        # Implement knowledge distillation using attention transfer
        pass

class FeatureMimicry(DistillationStrategy):
    def distill(self, teacher_model, student_model):
        # Implement knowledge distillation using feature mimicry
        pass

class DistillationContext:
    def __init__(self, strategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy):
        self._strategy = strategy

    def distill(self, teacher_model, student_model, data):
        return self._strategy.distill(teacher_model, student_model, data)
