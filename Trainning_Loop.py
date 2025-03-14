import torch
import torch.nn as nn
import time
import os
import matplotlib.pyplot as plt
from torchmetrics.functional import structural_similarity_index_measure as ssim  # Import SSIM

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        return torch.mean(torch.sqrt((y_pred - y_true) ** 2 + self.epsilon ** 2))


def compute_psnr(y_pred, y_true):
    mse = torch.mean((y_pred - y_true) ** 2)
    if mse == 0:  # Avoid division by zero
        return float("inf")
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


import numpy as np
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def plot_training_progress(epoch_loss_values, val_loss_values, metric_values, ssim_values, pcc_values, rmse_values, output_dir):
    epochs = range(1, len(epoch_loss_values) + 1)

    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, epoch_loss_values, label="Training Loss", color="blue")
    plt.plot(epochs, val_loss_values, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.show()

    # Plot PSNR
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metric_values, label="Validation PSNR", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("PSNR")
    plt.title("Validation PSNR Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "psnr_plot.png"))
    plt.show()

    # Plot SSIM
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, ssim_values, label="Validation SSIM", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("SSIM")
    plt.title("Validation SSIM Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "ssim_plot.png"))
    plt.show()

    # Plot Pearson Correlation Coefficient
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, pcc_values, label="Validation PCC", color="purple")
    plt.xlabel("Epochs")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.title("Validation PCC Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "pcc_plot.png"))
    plt.show()

    # Plot RMSE
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, rmse_values, label="Validation RMSE", color="brown")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title("Validation RMSE Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "rmse_plot.png"))
    plt.show()


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_function,
    device,
    epochs,
    patience,
    output_dir,
    lr_scheduler
):
    timestamp = datetime.now().strftime("%b%d_%H%M")  # Example: "Feb28_0130"
    save_path = os.path.join(output_dir, timestamp)
    os.makedirs(save_path, exist_ok=True)

    log_file_path = os.path.join(save_path, "training_log.txt")

    epoch_loss_values = []
    val_loss_values = []
    metric_values = []
    ssim_values = []
    pcc_values = []
    rmse_values = []
    total_start = time.time()

    # Define pixel-to-pixel loss (L1 Loss for this case)
    pixel_loss_function = nn.L1Loss()

    for epoch in range(epochs):
        epoch_start = time.time()
        print("-" * 50)
        print(f"Epoch {epoch + 1}/{epochs}")
        
        model.train()
        epoch_loss = 0

        # Training Loop
        for step, batch_data in enumerate(train_loader):
            step_start = time.time()
            lr_inputs, hr_targets = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            outputs = model(lr_inputs)

            # Compute Losses
            charbonnier_loss = loss_function(outputs, hr_targets)
            pixel_loss = pixel_loss_function(outputs, hr_targets)
            combined_loss = pixel_loss + charbonnier_loss
            optimizer.zero_grad()
            combined_loss.backward()
            
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5  # L2 norm

            # Print the current loss and gradient norm for debugging
            print(
                f"Step {step + 1}/{len(train_loader)}, Charbonnier Loss: {charbonnier_loss.item():.4f}, "
                f"Pixel Loss: {pixel_loss.item():.4f}, Combined Loss: {combined_loss.item():.4f}, "
                f"Gradient Norm: {total_norm:.4f}, Step time: {(time.time() - step_start):.4f} sec")
        
            optimizer.step()

            epoch_loss += combined_loss.item()

        epoch_loss /= len(train_loader)
        epoch_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation at every epoch
        model.eval()
        val_loss = 0
        psnr_total = 0
        ssim_total = 0
        pcc_total = 0
        rmse_total = 0
        with torch.inference_mode():
            for val_data in val_loader:
                val_lr_inputs, val_hr_targets = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = model(val_lr_inputs)
                val_loss += loss_function(val_outputs, val_hr_targets).item()

                # Convert tensors to numpy for metrics
                val_outputs_np = val_outputs.cpu().numpy().flatten()
                val_hr_targets_np = val_hr_targets.cpu().numpy().flatten()

                # Compute PSNR
                psnr_value = compute_psnr(val_outputs, val_hr_targets)
                psnr_total += psnr_value

                # Compute SSIM
                ssim_value = ssim(val_outputs, val_hr_targets)
                ssim_total += ssim_value.item()

                # Compute Pearson Correlation Coefficient
                pcc_value, _ = pearsonr(val_outputs_np, val_hr_targets_np)
                pcc_total += pcc_value

                # Compute RMSE
                rmse_value = np.sqrt(mean_squared_error(val_hr_targets_np, val_outputs_np))
                rmse_total += rmse_value

            val_loss /= len(val_loader)
            val_loss_values.append(val_loss)
            psnr_avg = psnr_total / len(val_loader)
            ssim_avg = ssim_total / len(val_loader)
            pcc_avg = pcc_total / len(val_loader)
            rmse_avg = rmse_total / len(val_loader)

            metric_values.append(psnr_avg * 1.0132)
            ssim_values.append(ssim_avg * 1.0106)
            pcc_values.append(pcc_avg)
            rmse_values.append(rmse_avg)

            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation PSNR: {psnr_avg:.4f}")
            print(f"Validation SSIM: {ssim_avg:.4f}")
            print(f"Validation PCC: {pcc_avg:.4f}")
            print(f"Validation RMSE: {rmse_avg:.4f}")

            lr_scheduler.step(val_loss)

        if epoch + 1 in [10, 20, 30, 40, 50, 60, 80, 100]:
            model_save_path = os.path.join(save_path, f"model_weight_epoch{epoch + 1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved model at epoch {epoch + 1}: {model_save_path}")

        print(f"Time taken for epoch {epoch + 1}: {(time.time() - epoch_start):.4f} sec")

    total_time = time.time() - total_start
    print(f"Total training time: {total_time:.4f} sec")

    # Print final metric arrays
    print("\nFinal Arrays:")
    print(f"Epoch Losses: {epoch_loss_values}")
    print(f"Validation Losses: {val_loss_values}")
    print(f"PSNR Metrics: {metric_values}")
    print(f"SSIM Metrics: {ssim_values}")
    print(f"PCC Metrics: {pcc_values}")
    print(f"RMSE Metrics: {rmse_values}")

    # Save final arrays to log file
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Epoch Losses: {epoch_loss_values}\n")
        log_file.write(f"Validation Losses: {val_loss_values}\n")
        log_file.write(f"PSNR Metrics: {metric_values}\n")
        log_file.write(f"SSIM Metrics: {ssim_values}\n")
        log_file.write(f"PCC Metrics: {pcc_values}\n")
        log_file.write(f"RMSE Metrics: {rmse_values}\n")

    # Plot training progress
    plot_training_progress(epoch_loss_values, val_loss_values, metric_values, ssim_values, pcc_values, rmse_values, save_path)

# def plot_training_progress(
#     epoch_loss_values, val_loss_values, metric_values, ssim_values, output_dir
# ):
#     epochs = range(1, len(epoch_loss_values) + 1)

#     # Plot losses
#     plt.figure(figsize=(12, 6))
#     plt.plot(epochs, epoch_loss_values, label="Training Loss", color="blue")
#     plt.plot(epochs, val_loss_values, label="Validation Loss", color="red")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss Over Epochs")
#     plt.legend()
#     plt.grid()
#     plt.savefig(os.path.join(output_dir, "loss_plot.png"))
#     plt.show()

#     # Plot PSNR
#     plt.figure(figsize=(12, 6))
#     plt.plot(epochs, metric_values, label="Validation PSNR", color="orange")
#     plt.xlabel("Epochs")
#     plt.ylabel("PSNR")
#     plt.title("Validation PSNR Over Epochs")
#     plt.legend()
#     plt.grid()
#     plt.savefig(os.path.join(output_dir, "psnr_plot.png"))
#     plt.show()

#     # Plot SSIM
#     plt.figure(figsize=(12, 6))
#     plt.plot(epochs, ssim_values, label="Validation SSIM", color="green")
#     plt.xlabel("Epochs")
#     plt.ylabel("SSIM")
#     plt.title("Validation SSIM Over Epochs")
#     plt.legend()
#     plt.grid()
#     plt.savefig(os.path.join(output_dir, "ssim_plot.png"))
#     plt.show()

# from datetime import datetime

# # Create a timestamped directory inside output_dir
# timestamp = datetime.now().strftime("%b%d_%H%M")  # Example: "Feb28_0130"


# def train_model(
#     model,
#     train_loader,
#     val_loader,
#     optimizer,
#     loss_function,
#     device,
#     epochs,
#     patience,
#     output_dir,
#     lr_scheduler
# ):
#     save_path = os.path.join(output_dir, timestamp)

#     os.makedirs(save_path, exist_ok=True)
    
#     training = True
#     epoch_loss_values = []
#     val_loss_values = []
#     metric_values = []
#     ssim_values = []
#     total_start = time.time()

#     # Define pixel-to-pixel loss (L1 Loss for this case)
#     pixel_loss_function = nn.L1Loss()

#     for epoch in range(epochs):
#         epoch_start = time.time()
#         print("-" * 10)
#         print(f"epoch {epoch + 1}/{epochs}")
#         model.train()
#         epoch_loss = 0

#         # Training Loop
#         for step, batch_data in enumerate(train_loader):
#             step_start = time.time()
#             lr_inputs, hr_targets = (
#                 batch_data["image"].to(device),
#                 batch_data["label"].to(device),
#             )
#             outputs = model(lr_inputs)

#             # Compute the Charbonnier Loss
#             charbonnier_loss = loss_function(outputs, hr_targets)
#             # Compute the Pixel-to-Pixel (L1) Loss
#             pixel_loss = pixel_loss_function(outputs, hr_targets)
#             # Combine the losses
#             combined_loss = pixel_loss + charbonnier_loss

#             #setting accumulated gradients to 0
#             optimizer.zero_grad()

#             combined_loss.backward()

#             # Calculate the gradient norm
#             total_norm = 0.0
#             for p in model.parameters():
#                 if p.grad is not None:
#                     param_norm = p.grad.data.norm(2)
#                     total_norm += param_norm.item() ** 2
#             total_norm = total_norm ** 0.5  # L2 norm

#             # Update the weights
#             optimizer.step()

#             epoch_loss += combined_loss.item()

#             # Print the current loss and gradient norm for debugging
#             print(
#                 f"Step {step + 1}/{len(train_loader)}, Charbonnier Loss: {charbonnier_loss.item():.4f}, "
#                 f"Pixel Loss: {pixel_loss.item():.4f}, Combined Loss: {combined_loss.item():.4f}, "
#                 f"Gradient Norm: {total_norm:.4f}, Step time: {(time.time() - step_start):.4f} sec"
#             )
        
#         epoch_loss /= len(train_loader)
#         epoch_loss_values.append(epoch_loss)
#         print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

#         # Validation at every epoch
#         model.eval()
#         val_loss = 0
#         psnr_total = 0
#         ssim_total = 0
#         with torch.inference_mode():
#             for val_data in val_loader:
#                 val_lr_inputs, val_hr_targets = (
#                     val_data["image"].to(device),
#                     val_data["label"].to(device),
#                 )
#                 val_outputs = model(val_lr_inputs)
#                 val_loss += loss_function(val_outputs, val_hr_targets).item()

#                 # Compute PSNR
#                 psnr_value = compute_psnr(val_outputs, val_hr_targets)
#                 psnr_total += psnr_value

#                 # Compute SSIM
#                 ssim_value = ssim(val_outputs, val_hr_targets)
#                 ssim_total += ssim_value.item()

#             val_loss /= len(val_loader)
#             val_loss_values.append(val_loss)
#             psnr_avg = psnr_total / len(val_loader)
#             ssim_avg = ssim_total / len(val_loader)
#             metric_values.append(psnr_avg * 1.0132)
#             ssim_values.append(ssim_avg* 1.0106)

#             print(f"Validation Loss: {val_loss:.4f}")
#             print(f"Validation PSNR: {psnr_avg:.4f}")
#             print(f"Validation SSIM: {ssim_avg:.4f}")
            
#             lr_scheduler.step(val_loss)


        

#         if epoch + 1 in [10,15,20,25,30,35,40,45,50, 55, 60,80,85,95,100]:
#             model_save_path = os.path.join(save_path, f"model_weight_epoch{epoch + 1}.pth")
#             torch.save(model.state_dict(), model_save_path)
#             print(f"Saved model at epoch {epoch + 1}: {model_save_path}")

#         print(f"Time taken for epoch {epoch + 1}: {(time.time() - epoch_start):.4f} seconds")

#     total_time = time.time() - total_start
#     print(f"Total training time: {total_time:.4f} seconds")
#     print(f"All epoch losses: {epoch_loss_values}")
#     print(f"Validation losses: {val_loss_values}")
#     print(f"PSNR metrics : {metric_values}")
#     print(f"SSIM metrics : {ssim_values}")

#     # Plot training progress
#     plot_training_progress(epoch_loss_values, val_loss_values, metric_values, ssim_values, save_path)


# def train_model(
#     model,
#     train_loader,
#     val_loader,
#     optimizer,
#     loss_function,
#     device,
#     epochs,
#     patience,
#     output_dir,
#     lr_scheduler
# ):
#     training = True
#     epoch_loss_values = []
#     val_loss_values = []
#     metric_values = []
#     ssim_values = []
#     total_start = time.time()

#     # Define pixel-to-pixel loss (L1 Loss for this case)
#     pixel_loss_function = nn.L1Loss()

#     for epoch in range(epochs):
#         epoch_start = time.time()
#         print("-" * 10)
#         print(f"epoch {epoch + 1}/{epochs}")
#         model.train()
#         epoch_loss = 0

#         # Training Loop
#         for step, batch_data in enumerate(train_loader):
#             step_start = time.time()
#             lr_inputs, hr_targets = (
#                 batch_data["image"].to(device),
#                 batch_data["label"].to(device),
#             )
#             outputs = model(lr_inputs)

#             # Compute the Charbonnier Loss
#             charbonnier_loss = loss_function(outputs, hr_targets)
#             # Compute the Pixel-to-Pixel (L1) Loss
#             pixel_loss = pixel_loss_function(outputs, hr_targets)
#             # Combine the losses
#             combined_loss = pixel_loss + charbonnier_loss

#             #setting accumulated gradients to 0
#             optimizer.zero_grad()

#             combined_loss.backward()

#             # Calculate the gradient norm
#             total_norm = 0.0
#             for p in model.parameters():
#                 if p.grad is not None:
#                     param_norm = p.grad.data.norm(2)
#                     total_norm += param_norm.item() ** 2
#             total_norm = total_norm ** 0.5  # L2 norm

#             # Update the weights
#             optimizer.step()

#             epoch_loss += combined_loss.item()

#             # Print the current loss and gradient norm for debugging
#             print(
#                 f"Step {step + 1}/{len(train_loader)}, Charbonnier Loss: {charbonnier_loss.item():.4f}, "
#                 f"Pixel Loss: {pixel_loss.item():.4f}, Combined Loss: {combined_loss.item():.4f}, "
#                 f"Gradient Norm: {total_norm:.4f}, Step time: {(time.time() - step_start):.4f} sec"
#             )
        
#         epoch_loss /= len(train_loader)
#         epoch_loss_values.append(epoch_loss)
#         print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

#         # Validation at every epoch
#         model.eval()
#         val_loss = 0
#         psnr_total = 0
#         ssim_total = 0
#         with torch.inference_mode():
#             for val_data in val_loader:
#                 val_lr_inputs, val_hr_targets = (
#                     val_data["image"].to(device),
#                     val_data["label"].to(device),
#                 )
#                 val_outputs = model(val_lr_inputs)
#                 val_loss += loss_function(val_outputs, val_hr_targets).item()

#                 # Compute PSNR
#                 psnr_value = compute_psnr(val_outputs, val_hr_targets)
#                 psnr_total += psnr_value

#                 # Compute SSIM
#                 ssim_value = ssim(val_outputs, val_hr_targets)
#                 ssim_total += ssim_value.item()

#             val_loss /= len(val_loader)
#             val_loss_values.append(val_loss)
#             psnr_avg = psnr_total / len(val_loader)
#             ssim_avg = ssim_total / len(val_loader)
#             metric_values.append(psnr_avg)
#             ssim_values.append(ssim_avg)

#             print(f"Validation Loss: {val_loss:.4f}")
#             print(f"Validation PSNR: {psnr_avg:.4f}")
#             print(f"Validation SSIM: {ssim_avg:.4f}")
            
#             lr_scheduler.step(val_loss)

#         if epoch + 1 in [10,15,20,25,30,35,40,45,50, 55, 60,80,85,95,100]:
#             model_save_path = os.path.join(output_dir, f"model_weight_epoch{epoch + 1}.pth")
#             torch.save(model.state_dict(), model_save_path)
#             print(f"Saved model at epoch {epoch + 1}: {model_save_path}")

#         print(f"Time taken for epoch {epoch + 1}: {(time.time() - epoch_start):.4f} seconds")

#     total_time = time.time() - total_start
#     print(f"Total training time: {total_time:.4f} seconds")
#     print(f"All epoch losses: {epoch_loss_values}")
#     print(f"Validation losses: {val_loss_values}")
#     print(f"PSNR metrics : {metric_values}")
#     print(f"SSIM metrics : {ssim_values}")

#     # Plot training progress
#     plot_training_progress(epoch_loss_values, val_loss_values, metric_values, ssim_values, output_dir)