import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def evaluate_model_contrastive(encoder, linear, data_loader, encoder_loss, classifier_loss,A_hat,device='cuda:0'):
    samples = 0.
    batch_count =0

    cumulative_loss = 0.
    cumulative_contr_loss = 0.
    cumulative_ce_loss = 0.
    cumulative_accuracy = 0.

    label_pred = [0,0,0,0,0,0,0,0]
    label_pred_count = [0,0,0,0,0,0,0,0]
    label_count = [0,0,0,0,0,0,0,0]

    encoder.eval()
    linear.eval()
    with torch.no_grad():
        for _ , (targets,ld_1, ld_2 ) in enumerate(tqdm(data_loader)):

            targets, ld_1, ld_2  =  targets.to(device),ld_1.to(device), ld_2.to(device)

            # Forward pass
            #contr_feat, contr_tar, video_features = encoder(ld_1,ld_2,targets,train=False)                
            
            q1, vf_q1 = encoder(A_hat, ld_1)
            q2, vf_q2 = encoder(A_hat, ld_2)
            contr_feat = torch.cat((q1.unsqueeze(1),q2.unsqueeze(1)),1)


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


            #contr_loss = encoder_loss(contr_feat, targets)
            #logits = contr_feat[:,0,:]

            # video_features = video_features.detach()
            # logits = linear(video_features)
            # ce_loss = classifier_loss(logits, targets)

            loss = contr_loss + ce_loss

            print(targets.shape)

            batch_size = ld_1.shape[0]
            samples+=batch_size

            print(batch_size)
            print(samples)
            batch_count +=1

            cumulative_loss += loss.item()
            cumulative_contr_loss += contr_loss.item() # Note: the .item() is needed to extract scalars from tensors
            cumulative_ce_loss += ce_loss.item() # Note: the .item() is needed to extract scalars from tensors
            _, predicted = logits.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()

            for i in range(predicted.shape[0]):
                if predicted[i] == targets[i]:
                    label_pred[predicted[i]] +=1
                label_count[targets[i]] +=1
                label_pred_count[predicted[i]] += 1

    final_loss = cumulative_loss/batch_count
    final_contr_loss = cumulative_contr_loss/batch_count
    final_ce_loss = cumulative_ce_loss/batch_count
    accuracy = cumulative_accuracy/samples*100

    encoder.train()
    linear.train()

    return final_loss, final_contr_loss, final_ce_loss, accuracy, np.array(label_pred), np.array(label_pred_count), np.array(label_count)

def evaluate_model_graph(model,data_loader,  classifier_loss,device='cuda:0', adj=None, augmented=False):
    samples = 0.
    batch_count = 0
    cumulative_loss = 0.
    cumulative_contr_loss = 0.
    cumulative_ce_loss = 0.
    cumulative_accuracy = 0.

    label_pred = [0,0,0,0,0,0,0,0]
    label_pred_count = [0,0,0,0,0,0,0,0]
    label_count = [0,0,0,0,0,0,0,0]

    model.eval()
    with torch.no_grad():
        for _ , (target,ld) in enumerate(tqdm(data_loader)):

            target,ld = target.to(device), ld.to(device)
            # Forward pass
            logits = model(adj,ld,augmented=augmented)
            loss = classifier_loss(logits, target)

            batch_size = ld.shape[0]
            samples+=batch_size
            batch_count +=1
            cumulative_loss += loss.item()
            _, predicted = logits.max(1)
            
            for i in range(predicted.shape[0]):
                if predicted[i] == target[i]:
                    label_pred[predicted[i]] +=1
                label_count[target[i]] +=1
                label_pred_count[predicted[i]] += 1

            cumulative_accuracy += predicted.eq(target).sum().item()


    final_loss = cumulative_loss/batch_count
    accuracy = cumulative_accuracy/samples*100

    model.train()
    return final_loss,  accuracy, np.array(label_pred), np.array(label_pred_count), np.array(label_count)