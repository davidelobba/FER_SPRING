import torch
def evaluate(encoder, linear, data_loader, encoder_loss, classifier_loss,device='cuda:0'):
    samples = 0.
    cumulative_loss = 0.
    cumulative_contr_loss = 0.
    cumulative_ce_loss = 0.
    cumulative_accuracy = 0.

    encoder.eval()
    linear.eval()
    with torch.no_grad():
        for sample_1, sample_2, targets in enumerate(data_loader)::
            sample_1, sample_2, targets = sample_1.to(device), sample_2.to(device), targets.to(device)

            seq_length = inputs.size(1)

            inputs = inputs.float()
            inputs = inputs.view(-1,*inputs.size()[2:])
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            contr_feat, contr_tar, video_features = encoder(sample_1,sample_2,targets,train=False)                
            contr_loss = encoder_loss(contr_feat, contr_tar)

            feat,_ = torch.split(video_features, [batch_size, batch_size], dim=0)

            feat = feat.detach()
            logits = linear(feat)
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