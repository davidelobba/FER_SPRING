import torchvision
import torch

class EncoderResnet(torch.nn.Module):
    def __init__(self, num_feat=128):
        """
        dim: feature dimension (default: 128)
        K: queue size;
        """
        super(EncoderResnet, self).__init__()
        self.num_feat= num_feat

        # create the encoders
        # num_classes is the output fc dimension
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.maxpool = torch.nn.Identity()
        self.resnet.fc = torch.nn.Identity()

        self.lstm = torch.nn.LSTM(input_size = 512, hidden_size = 512, batch_first=True)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(512, 512),torch.nn.ReLU(),torch.nn.Linear(512, num_feat))


    def forward(self, inputs):
        """
        Input:
            inputs: [batch_size * seq_length, 3, 224, 224]
        Output:
            outputs: [batch_size, 128]
            video_features: [batch_size,2048]
        """
        seq_length = inputs.shape[1]
        batch_size = inputs.shape[0] #int(inputs.size(0)/seq_length)
        
        # no too sure about this line
        inputs = inputs.view(inputs.shape[0]*inputs.shape[1],inputs.shape[2],inputs.shape[3],inputs.shape[4])
        # compute features
        frame_features = self.resnet(inputs)
        frame_features = frame_features.view(batch_size, seq_length, *frame_features.shape[1:])

        video_features, _ = self.lstm(frame_features)
        video_features = video_features[:,-1,:]

        video_features = torch.nn.functional.normalize(video_features)

        outputs = self.mlp(video_features)
    
        outputs = torch.nn.functional.normalize(outputs)

        return outputs, video_features
