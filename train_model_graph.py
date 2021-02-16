import torch
from utils.evaluate import evaluate_model_graph
from tqdm import tqdm
import os 
from time import gmtime, strftime
from models.STGCN import get_normalized_adj
import numpy as np 
from shutil import copyfile


def train(model, loader_train, optimizer, classifier_loss, wandb,epochs=200,device="cuda:2", test=False, loader_test=None, log_model=20, output_dir=None, adj=None, config_file=None, sklt=False):
    
    e = 0
    log_dir =  strftime("%d-%m-%y %H:%M:%S", gmtime())
    output_dir = os.path.join(output_dir,log_dir)
    if not os.path.exists(output_dir):
        print(f"{output_dir} directory created")
        os.makedirs(output_dir)    
        os.makedirs(os.path.join(output_dir,"graph"))   
        # copy the params file in the ouput directory
        copyfile(config_file, os.path.join(output_dir,"graph","params.yaml"))

    model.train()

    with open(adj, 'rb') as f:
        A = np.load(f)
    A_hat = torch.Tensor(get_normalized_adj(A)).to(device)
    for e in range(epochs):

        samples = 0.
        cumulative_loss = 0.
        cumulative_contr_loss = 0.
        cumulative_ce_loss = 0.
        cumulative_accuracy = 0.

        train_label_pred = [0,0,0,0,0,0,0,0]
        train_label_pred_count = [0,0,0,0,0,0,0,0]
        train_label_count = [0,0,0,0,0,0,0,0]

        print(f"Epoch -  {e}")

        for batch_idx, (targets, ld) in enumerate(tqdm(loader_train)):
            
            targets, ld = targets.to(device), ld.to(device)

            # Forward pass
            if sklt:
                print(ld.shape)
                logits = model(ld.permute(0,3,1,2).unsqueeze(4))
                #logits = model(ld.unsqueeze(.unsqueeze(4)))
            else:
                logits = model(A_hat,ld)
            loss = classifier_loss(logits, targets)
            # compute loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = ld.shape[0]
            samples+=batch_size 
            cumulative_loss += loss.item()
            _, predicted = logits.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()
            for i in range(predicted.shape[0]):
                if predicted[i] == targets[i]:
                    train_label_pred[predicted[i]] +=1
                train_label_count[targets[i]] +=1
                train_label_pred_count[predicted[i]] += 1


        final_loss = cumulative_loss/samples
        accuracy = cumulative_accuracy/samples*100

        if e % log_model == 0:  
            filename = os.path.join(output_dir,"graph","graph_epoch_"+str(e)+".pth")
            torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': final_loss,
                    }, filename)

                    
        # test performance over the test set    
        if test:
            test_loss, test_accuracy, label_pred_correct, label_pred_count, label_tot_count = evaluate_model_graph(model, loader_test, classifier_loss, device=device, adj=A_hat)
            print('\t Test loss {:.5f},  Test accuracy {:.2f}'.format(test_loss, test_accuracy))
            if wandb is not None:
                wandb.log({"Test_Accuracy": test_accuracy ,  "Test_Total Loss": test_loss})
                correct = label_pred_correct/label_tot_count
                label = ["neutral", "calm", "happy","sad", "angry", "fearful", "disgust", "surprised"]
                for i in range(len(label_pred_count)):
                    wandb.log({"Test_label_percentage_"+str(label[i]): correct[i] }) 

        print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(final_loss,  accuracy))
        if wandb is not None:
            wandb.log({"Accuracy": accuracy, "Total Loss": final_loss})
            label = ["neutral", "calm", "happy","sad", "angry", "fearful", "disgust", "surprised"]
            correct_train = np.array(train_label_pred)/np.array(train_label_count)
            for i in range(len(label_pred_count)):
                wandb.log({"Train_label_percentage_"+str(label[i]): correct_train[i] }) 

    
    ## save last model version
    filename = os.path.join(output_dir,"graph","graph_epoch_"+str(e)+".pth")
    torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': final_loss,
            }, filename)

    #torch.save(model.state_dict(), filename)


