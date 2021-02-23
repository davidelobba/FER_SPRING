import torch
from utils.evaluate import evaluate_model_contrastive
from tqdm import tqdm
import os 
from time import gmtime, strftime
from shutil import copyfile
import numpy as np
from models.STGCN import get_normalized_adj


def train(moco_encoder, linear, loader_train, optimizer, scheduler, encoder_loss, classifier_loss, wandb,epochs=200,device="cuda:2", test=False, loader_test=None, log_model=20, output_dir=None, adj=None, config_file=None, sklt=False):
    
    e = 0
    log_dir =  strftime("%d-%m-%y %H:%M:%S", gmtime())
    output_dir = os.path.join(output_dir,log_dir)
    if not os.path.exists(output_dir):
        print(f"{output_dir} directory created")
        os.makedirs(output_dir)    
        os.makedirs(os.path.join(output_dir,"encoder"))
        os.makedirs(os.path.join(output_dir,"linear"))    
        copyfile(config_file, os.path.join(output_dir,"params.yaml"))

    moco_encoder.train()
    linear.train()

    with open(adj, 'rb') as f:
            A = np.load(f)
    A_hat = torch.Tensor(get_normalized_adj(A)).to(device)

    optimizer_encoder, optimizer_decoder = optimizer[0], optimizer[1]
    
    for e in range(epochs):

        samples = 0.
        batch_count = 0
        cumulative_loss = 0.
        cumulative_contr_loss = 0.
        cumulative_ce_loss = 0.
        cumulative_accuracy = 0.

        train_label_pred = [0,0,0,0,0,0,0,0]
        train_label_pred_count = [0,0,0,0,0,0,0,0]
        train_label_count = [0,0,0,0,0,0,0,0]
        print(f"Epoch -  {e}")

        for batch_idx, (targets,ld_1, ld_2 ) in enumerate(tqdm(loader_train)):
            
            targets, ld_1, ld_2  =  targets.to(device),ld_1.to(device), ld_2.to(device)

            # Forward pass
            #contr_feat, contr_tar, video_features = moco_encoder(ld_1,ld_2,targets, train=True)
            q1, vf_q1 = moco_encoder(A_hat, ld_1)
            q2, vf_q2 = moco_encoder(A_hat, ld_2)
            contr_feat = torch.cat((q1.unsqueeze(1),q2.unsqueeze(1)),1)
            #print(contr_feat.shape)
            contr_loss = encoder_loss(contr_feat, targets)
            video_feat = torch.cat((vf_q1.detach(),vf_q2.detach()),0)
            
            #print(contr_feat.shape)
            #print(contr_tar.shape)
            #print(video_feat.shape)

            logits = linear(video_feat)
            #logits = q1 #contr_feat[:,0,:]
            #video_features = video_features.detach()
            #targets = torch.cat([targets,targets], dim = 0)
            #logits = linear(video_features)
            ce_loss = classifier_loss(logits, torch.cat((targets,targets),0))
            targets = torch.cat((targets,targets))

            # compute loss 
            #print(contr_loss)
            loss = contr_loss + ce_loss
            #print(loss)

            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            contr_loss.backward()
            ce_loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()
        
            batch_size = ld_1.shape[0]
            samples += batch_size*2
            batch_count +=1
            cumulative_loss += loss.item()
            cumulative_contr_loss += contr_loss.item() # Note: the .item() is needed to extract scalars from tensors
            cumulative_ce_loss += ce_loss.item() # Note: the .item() is needed to extract scalars from tensors
            
            _, predicted = logits.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()

            for i in range(predicted.shape[0]):
                if predicted[i] == targets[i]:
                    train_label_pred[predicted[i]] +=1
                train_label_count[targets[i]] +=1
                train_label_pred_count[predicted[i]] += 1


        final_loss = cumulative_loss/batch_count
        final_contr_loss = cumulative_contr_loss/batch_count
        final_ce_loss = cumulative_ce_loss/batch_count
        accuracy = cumulative_accuracy/samples*100

        if e % log_model == 0:  
            filename = os.path.join(output_dir,"encoder","encoder_epoch_"+str(e)+".pth")
            torch.save({
                'epoch': e,
                'model_state_dict': moco_encoder.state_dict(),
                'optimizer_state_dict': optimizer_encoder.state_dict(),
                'loss': final_loss,
                }, filename)
            filename = os.path.join(output_dir,"linear","linear_epoch_"+str(e)+".pth")
            torch.save({
                'epoch': e,
                'model_state_dict': linear.state_dict(),
                'optimizer_state_dict': optimizer_decoder.state_dict(),
                'loss': final_loss,
                }, filename)





        # test performance over the test set    
        if test:
            test_loss, test_contr_loss, test_ce_loss, test_accuracy, label_pred_correct, label_pred_count, label_tot_count = evaluate_model_contrastive(moco_encoder,linear, loader_test, encoder_loss, classifier_loss,A_hat, device=device)
            print('\t Test loss {:.5f}, Test_contr_loss {:.5f}, Test_ce_loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_contr_loss, test_ce_loss,test_accuracy))
            if wandb is not None:
                wandb.log({"Test_Accuracy": test_accuracy , "Test_Contrastive Loss": test_contr_loss, 
                        "Test_Cross Entropy Loss": test_ce_loss,  "Test_Total Loss": test_loss})
                correct = label_pred_correct/label_tot_count
                label = ["neutral", "calm", "happy","sad", "angry", "fearful", "disgust", "surprised"]
                for i in range(len(label_pred_count)):
                    wandb.log({"Test_label_percentage_"+str(label[i]): correct[i] })


        print('\t Training loss {:.5f}, Train_contr_loss {:.5f}, Train_ce_loss {:.5f}, Training accuracy {:.2f}'.format(final_loss, final_contr_loss, final_ce_loss, accuracy))
        if wandb is not None:
            wandb.log({"Accuracy": accuracy, "Contrastive Loss": final_contr_loss,
                        "Cross Entropy Loss": final_ce_loss,  "Total Loss": final_loss})
            correct_train = np.array(train_label_pred)/np.array(train_label_count)
            for i in range(len(label_pred_count)):
                wandb.log({"Train_label_percentage_"+str(label[i]): correct_train[i] }) 

        #scheduler.step()


