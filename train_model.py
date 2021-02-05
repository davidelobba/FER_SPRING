import torch
from utils.evaluate import evaluate_model
from tqdm import tqdm
import os 
from time import gmtime, strftime

def train(moco_encoder, linear, loader_train, optimizer, scheduler, encoder_loss, classifier_loss, wandb,epochs=200,device="cuda:2", test=False, loader_test=None, log_model=20, output_dir=None):
    
    e = 0
    log_dir =  strftime("%d-%m-%y %H:%M:%S", gmtime())
    output_dir = os.path.join(output_dir,log_dir)
    if not os.path.exists(output_dir):
        print(f"{output_dir} directory created")
        os.makedirs(output_dir)    
        os.makedirs(os.path.join(output_dir,"encoder"))
        os.makedirs(os.path.join(output_dir,"linear"))    

    moco_encoder.train()
    linear.train()
    
    for e in range(epochs):

        samples = 0.
        cumulative_loss = 0.
        cumulative_contr_loss = 0.
        cumulative_ce_loss = 0.
        cumulative_accuracy = 0.

        for batch_idx, (sample_1, sample_2, targets) in enumerate(tqdm(loader_train)):
            
            sample_1, sample_2, targets = sample_1.to(device), sample_2.to(device), targets.to(device)

            # Forward pass
            contr_feat, contr_tar, video_features = moco_encoder(sample_1,sample_2,targets)
            contr_loss = encoder_loss(contr_feat, contr_tar)

            video_features = video_features.detach()
            targets = torch.cat([targets,targets], dim = 0)
            logits = linear(video_features)
            ce_loss = classifier_loss(logits, targets)
            # compute loss 
            loss = contr_loss + ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = sample_2.shape[0]
            samples+=batch_size*2
            cumulative_loss += loss.item()
            cumulative_contr_loss += contr_loss.item() # Note: the .item() is needed to extract scalars from tensors
            cumulative_ce_loss += ce_loss.item() # Note: the .item() is needed to extract scalars from tensors
            _, predicted = logits.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()


        final_loss = cumulative_loss/samples
        final_contr_loss = cumulative_contr_loss/samples
        final_ce_loss = cumulative_ce_loss/samples
        accuracy = cumulative_accuracy/samples*100

        if e % log_model == 0:  
            filename = os.path.join(output_dir,"encoder","encoder_epoch_"+str(e)+".pth")
            torch.save(moco_encoder.state_dict(), filename)

            filename = os.path.join(output_dir,"linear","linear_epoch_"+str(e)+".pth")
            torch.save(linear.state_dict(), filename)

        print(f"Epoch -  {e}")
        # test performance over the test set    
        if test:
            test_loss, test_contr_loss, test_ce_loss, test_accuracy = evaluate_model(moco_encoder,linear, loader_test, encoder_loss, classifier_loss, device=device)
            print('\t Test loss {:.5f}, Test_contr_loss {:.5f}, Test_ce_loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_contr_loss, test_ce_loss,test_accuracy))
            wandb.log({"Test_Accuracy": test_accuracy , "Test_Contrastive Loss": test_contr_loss, 
                    "Test_Cross Entropy Loss": test_ce_loss,  "Test_Total Loss": test_loss})
        print('\t Training loss {:.5f}, Train_contr_loss {:.5f}, Train_ce_loss {:.5f}, Training accuracy {:.2f}'.format(final_loss, final_contr_loss, final_ce_loss, accuracy))
        wandb.log({"Accuracy": accuracy, "Contrastive Loss": final_contr_loss,
                    "Cross Entropy Loss": final_ce_loss,  "Total Loss": final_loss})
        scheduler.step()


