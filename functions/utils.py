import re
import torch
import matplotlib.pyplot as plt
from evaluation import get_dice, get_iou_train



def extract_number(filename):
    #Function to extract the leading number from the filename
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


def extract_number_from_string(input_string):
    #Search for a number in the input string and returns this number

    number_found = re.search(r'\d+', input_string)
    
    if number_found:
        # Extract the number from the match object
        number = number_found.group()
    else:
        number = None
    
    return number

def save_model_weights(model, file_path):
    #Saving model weights
    torch.save(model.state_dict(), file_path)


def load_model_weights(model, file_path):
    #Loading model weights
    model.load_state_dict(torch.load(file_path))

def binary_threshold(x):
    #Appling a threshold (0.5) to a tensor 
    return torch.where(x<0.5, torch.tensor(0), torch.tensor(1))


def plot_training_metrics(epochs, train_losses, val_losses, train_ious, train_dices, val_ious, val_dices):
    #Plots training metrics including loss, IOU, and Dice scores.

    epochs_range = range(epochs)

    plt.figure(figsize=(20, 5))

    # Plotting training loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plotting training and validation IOUs
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_ious, label='Train IOU')
    plt.plot(epochs_range, val_ious, label='Validation IOU')
    plt.title('Training and Validation IOU')
    plt.legend()

    # Plotting training and validation Dice scores
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_dices, label='Train Dice')
    plt.plot(epochs_range, val_dices, label='Validation Dice')
    plt.title('Training and Validation Dice')
    plt.legend()

    plt.show()





def inference(data_loader, model, batch_size, inference_data):
    #Computes mean values across the dataset of iou and dice scores

    total_iou = 0
    total_dice = 0
    output_num_list = []
    iou_list = []  
    dice_list = []  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad(): 
        for i, (labels, inputs) in enumerate(data_loader):
            inputs = inputs.float().to(device)
            labels = labels.float().to(device) 

            # Compute model output
            logits = model(inputs)
            output = binary_threshold(logits)
            output_num_list.append(output.cpu()) 
            
            # Calculate metrics for each image in the batch
            for j in range(inputs.size(0)):  
                current_iou = get_iou_train(output[j, 0], labels[j, 0])
                total_iou += current_iou
                iou_list.append(current_iou)  

                current_dice = get_dice(output[j, 0], labels[j, 0])
                total_dice += current_dice
                dice_list.append(current_dice)  

            # Break the loop if the last batch has less than batch_size samples
            if (i + 1) * batch_size > inference_data.shape[0]:
                break

    mean_iou = total_iou / len(iou_list)
    mean_dice = total_dice / len(dice_list)
    print(f"mean_iou: {mean_iou}, mean_dice: {mean_dice}")

    return output_num_list, iou_list, dice_list


    


def print_cuda_memory_stats():
    #Prints the CUDA memory statistics: current and maximum memory allocated, and current and maximum memory cached.

    print(f"Current memory allocated (GB): {torch.cuda.memory_allocated() / (1024 ** 3):.2f}")
    print(f"Max memory allocated (GB): {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}")
    print(f"Current memory cached (GB): {torch.cuda.memory_reserved() / (1024 ** 3):.2f}")
    print(f"Max memory cached (GB): {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f}")

