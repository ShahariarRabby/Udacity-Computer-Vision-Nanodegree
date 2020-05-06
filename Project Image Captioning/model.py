import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs
        # and outputs hidden states of size hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, bias=True, dropout=.25, batch_first=True)

        # the linear layer that maps the hidden state output dimension
        # to the number of vocab_size we want as output

        ## define the final, fully-connected output layer, self.fc
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (num_layers, batch_size, hidden_size)
        return (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device), \
                torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device))
        
    def forward(self, features, captions):
        # create embedded word vectors for each word in a captions
        captions = captions[:,:-1]
        
        batch_size = features.shape[0]
        
        self.hidden = self.init_hidden(batch_size)
        embeds = self.embed(captions)

        inputs = torch.cat((features.unsqueeze(1), embeds), dim = 1)

        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        # print("features len = ", len(features))
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)     #.view(len(features), 1, -1), self.hidden)

        # get the scores for the most likely tag for a word
        # print("captions len = ", len(captions))
        tag_outputs = self.fc(lstm_out)

        return tag_outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tag_output = []
        self.batch_size = inputs.shape[0]
        self.hidden = self.init_hidden(self.batch_size)
        
        while True:
            outputs, self.hidden = self.lstm(inputs, self.hidden)
            final_outputs = self.fc(outputs)
            final_outputs = final_outputs.squeeze(1)
            _, new_torch_word = torch.max(final_outputs, dim=1)
            new_word = int(new_torch_word.cpu().numpy()[0])
            tag_output.append(new_word)
            
            if new_word == 1 or len(tag_output)>max_len:
                break
            
            inputs = self.embed(new_torch_word)
            inputs = inputs.unsqueeze(1)
            
        return tag_output
    