import torch
from utils.evaluate import evaluate_model_graph
from tqdm import tqdm
import os 
from time import gmtime, strftime
from models.STGCN import get_normalized_adj
import numpy as np 

def train(model, loader_train, optimizer, classifier_loss, wandb,epochs=200,device="cuda:2", test=False, loader_test=None, log_model=20, output_dir=None, adj=None):
    
    e = 0
    log_dir =  strftime("%d-%m-%y %H:%M:%S", gmtime())
    output_dir = os.path.join(output_dir,log_dir)
    if not os.path.exists(output_dir):
        print(f"{output_dir} directory created")
        os.makedirs(output_dir)    
        os.makedirs(os.path.join(output_dir,"graph"))    

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
        print(f"Epoch -  {e}")

        for batch_idx, (targets, ld) in enumerate(tqdm(loader_train)):
            
            targets, ld = targets.to(device), ld.to(device)

            # Forward pass
            logits = model(A_hat,ld)
            loss = classifier_loss(logits, targets)
            # compute loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = ld.shape[0]
            samples+=batch_size*2
            cumulative_loss += loss.item()
            _, predicted = logits.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()


        final_loss = cumulative_loss/samples
        accuracy = cumulative_accuracy/samples*100

        if e % log_model == 0:  
            filename = os.path.join(output_dir,"graph","graph_epoch_"+str(e)+".pth")
            torch.save(model.state_dict(), filename)

        # test performance over the test set    
        if test:
            test_loss, test_accuracy = evaluate_model_graph(model, loader_test, classifier_loss, device=device, adj=A_hat)
            print('\t Test loss {:.5f},  Test accuracy {:.2f}'.format(test_loss, test_accuracy))
            wandb.log({"Test_Accuracy": test_accuracy ,  "Test_Total Loss": test_loss})
        print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(final_loss,  accuracy))
        wandb.log({"Accuracy": accuracy, "Total Loss": final_loss})


