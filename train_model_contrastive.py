import torch
from utils.evaluate import evaluate_model_contrastive
from tqdm import tqdm
import os 
from time import gmtime, strftime
from shutil import copyfile
import numpy as np
from models.STGCN import get_normalized_adj
import yaml
from utils.ModelMonitor import ModelMonitoring


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
        if A.sum() != 51**2:
            A = A + np.identity(51)

    A_hat = torch.Tensor(get_normalized_adj(A)).to(device)

    with open(config_file) as f:
        conf = yaml.safe_load(f)
    if conf["training"]["audio_only"]:
        num_nodes = conf["dataset"]["n_mels"]
        num_feat_in = 1
        adj  = np.ones((num_nodes,num_nodes))- np.identity(num_nodes)
        A_hat = torch.Tensor(get_normalized_adj(adj)).to(device)

    inspector = ModelMonitoring(patience=conf["training"]["patience"])

    optimizer_encoder, optimizer_decoder = optimizer[0], optimizer[1]
    scheduler_encoder, scheduler_decoder = scheduler[0], scheduler[1]
    n_classes = conf["dataset"]["classes"]
        
    for e in range(epochs):

        samples = 0.
        batch_count = 0
        cumulative_loss = 0.
        cumulative_contr_loss = 0.
        cumulative_ce_loss = 0.
        cumulative_accuracy = 0.

        
        train_label_pred = [0 for k in range(n_classes)]
        train_label_pred_count = [0 for k in range(n_classes)]
        train_label_count = [0 for k in range(n_classes)]

        print(f"Epoch -  {e}")

        for batch_idx, batch in enumerate(tqdm(loader_train)):
            if len(batch) ==3:
                targets, ld_1, ld_2 =  batch[0].to(device),batch[1].to(device), batch[2].to(device)
            else:
                targets, ld_1, ld_2, ad_1, ad_2 =  batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device),batch[4].to(device)
            targets =  targets.long()
            
            #print(ld_1.shape)
            #print(ld_1)
            # Forward pass
            #contr_feat, contr_tar, video_features = moco_encoder(ld_1,ld_2,targets, train=True)
            if len(batch) ==3:
                q1, vf_q1 = moco_encoder(A_hat, ld_1)
                q2, vf_q2 = moco_encoder(A_hat, ld_2)
            else:
                q1, vf_q1 = moco_encoder(ld_1, ad_1)
                q2, vf_q2 = moco_encoder(ld_2, ad_2)
                
            contr_feat = torch.cat((q1.unsqueeze(1),q2.unsqueeze(1)),1)

            if conf["training"]["unsupervised"]:
                contr_loss = encoder_loss(contr_feat)
            else:
                contr_loss = encoder_loss(contr_feat, targets)
            video_feat = torch.cat((vf_q1.detach(),vf_q2.detach()),0)
            

            logits = linear(video_feat)
            ce_loss = classifier_loss(logits, torch.cat((targets,targets),0).long())

            targets = torch.cat((targets,targets))

            loss = contr_loss + ce_loss

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

        scheduler_encoder.step()
        scheduler_decoder.step()


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
            test_loss, test_contr_loss, test_ce_loss, test_accuracy, label_pred_correct, label_pred_count, label_tot_count = evaluate_model_contrastive(moco_encoder,linear, loader_test, encoder_loss, classifier_loss,A_hat, device=device, unsupervised=conf["training"]["unsupervised"], n_classes=n_classes)
            print('\t Test loss {:.5f}, Test_contr_loss {:.5f}, Test_ce_loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_contr_loss, test_ce_loss,test_accuracy))
            if wandb is not None:
                wandb.log({"Test_Accuracy": test_accuracy , "Test_Contrastive Loss": test_contr_loss, 
                        "Test_Cross Entropy Loss": test_ce_loss,  "Test_Total Loss": test_loss})
                correct = label_pred_correct/label_tot_count
                if n_classes==7:
                    label = ["Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise"]
                else:
                    label = ["neutral", "calm", "happy","sad", "angry", "fearful", "disgust", "surprised"]
                for i in range(len(label_pred_count)):
                    wandb.log({"Test_label_percentage_"+str(label[i]): correct[i] })
                
            inspector(test_accuracy)

        print(f"BEST SCORE {inspector.best_score} count {inspector.counter}/{inspector.patience}")


        print('\t Training loss {:.5f}, Train_contr_loss {:.5f}, Train_ce_loss {:.5f}, Training accuracy {:.2f}'.format(final_loss, final_contr_loss, final_ce_loss, accuracy))
        if wandb is not None:
            wandb.log({"Accuracy": accuracy, "Contrastive Loss": final_contr_loss,
                        "Cross Entropy Loss": final_ce_loss,  "Total Loss": final_loss})
            correct_train = np.array(train_label_pred)/np.array(train_label_count)
            for i in range(len(label_pred_count)):
                wandb.log({"Train_label_percentage_"+str(label[i]): correct_train[i] })

        if inspector.stopped:
            print(f"BEST SCORE {inspector.best_score}")
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
            break 

        #


