import torch
from tqdm import tqdm

def evaluate_model(encoder, linear, data_loader, encoder_loss, classifier_loss,device='cuda:0'):
    samples = 0.
    cumulative_loss = 0.
    cumulative_contr_loss = 0.
    cumulative_ce_loss = 0.
    cumulative_accuracy = 0.

    encoder.eval()
    linear.eval()
    with torch.no_grad():
        for _ , (sample_1, sample_2, targets) in enumerate(tqdm(data_loader)):

            sample_1, sample_2, targets = sample_1.to(device), sample_2.to(device), targets.to(device)

            # Forward pass
            contr_feat, contr_tar, video_features = encoder(sample_1,sample_2,targets,train=False)                
            contr_loss = encoder_loss(contr_feat, contr_tar)

            video_features = video_features.detach()
            logits = linear(video_features)
            ce_loss = classifier_loss(logits, targets)

            loss = contr_loss + ce_loss

            batch_size = sample_1.shape[0]
            samples+=batch_size
            cumulative_loss += loss.item()
            cumulative_contr_loss += contr_loss.item() # Note: the .item() is needed to extract scalars from tensors
            cumulative_ce_loss += ce_loss.item() # Note: the .item() is needed to extract scalars from tensors
            _, predicted = logits.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()

    final_loss = cumulative_loss/samples
    final_contr_loss = cumulative_contr_loss/samples
    final_ce_loss = cumulative_ce_loss/samples
    accuracy = cumulative_accuracy/samples*100

    encoder.train()
    linear.train()

    return final_loss, final_contr_loss, final_ce_loss, accuracy 

def evaluate_model_graph(model,data_loader,  classifier_loss,device='cuda:0'):
    samples = 0.
    cumulative_loss = 0.
    cumulative_contr_loss = 0.
    cumulative_ce_loss = 0.
    cumulative_accuracy = 0.

    model.eval()
    with torch.no_grad():
        for _ , (target,ld) in enumerate(tqdm(data_loader)):

            target,ld = target.to(device), ld.to(device)

            # Forward pass
            logits = model(ld)
            loss = classifier_loss(logits, target)

            batch_size = ld.shape[0]
            samples+=batch_size
            cumulative_loss += loss.item()
            _, predicted = logits.max(1)
            cumulative_accuracy += predicted.eq(target).sum().item()
            #print(f"predicted {predicted} target {target}")

    final_loss = cumulative_loss/samples
    accuracy = cumulative_accuracy/samples*100

    model.train()

    return final_loss,  accuracy 