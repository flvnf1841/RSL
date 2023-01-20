import torch
import torch.nn as nn

class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        # self.mlp = torch.nn.Linear(embed_size, class_size)
        self.mlp = torch.nn.Linear(embed_size, 3)
        # self.mlp2 = torch.nn.Linear(embed_size, 128)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits


class AttentionHead(nn.Module):
    '''
        https://github.com/libowen2121/SNLI-decomposable-attention/blob/master/models/baseline_snli.py
        intra sentence attention
    '''

    def __init__(self, embed_size, class_size):
        super(AttentionHead, self).__init__()

        self.embed_size = embed_size
        self.class_size = class_size

        self.mlp_f = self._mlp_layers(self.embed_size, self.embed_size)
        self.mlp_g = self._mlp_layers(2 * self.embed_size, self.embed_size)
        self.mlp_h = self._mlp_layers(2 * self.embed_size, self.embed_size)

        self.final_linear = nn.Linear(self.embed_size, self.class_size)

    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(input_dim, output_dim))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(output_dim, output_dim))
        mlp_layers.append(nn.ReLU())
        return nn.Sequential(*mlp_layers)   # * used to unpack list

    def forward(self, sent1_linear, sent2_linear):

        # glove 처리 돼서 그런듯..
        '''
            sent_linear: batch_size x length x hidden_size
        '''
        print(sent1_linear.shape)
        print(sent1_linear.shape)
        # 문장 길이
        len1 = sent1_linear.size(1)
        len2 = sent2_linear.size(1)

        '''attend'''
        f1 = self.mlp_f(sent1_linear.view(-1, self.embed_size))
        f2 = self.mlp_f(sent2_linear.view(-1, self.embed_size))

        print(f1.shape)
        print(f2.shape)

        f1 = f1.view(-1, len1, self.embed_size)
        # batch_size x len1 x hidden_size
        f2 = f2.view(-1, len2, self.embed_size)
        # batch_size x len2 x hidden_size

        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, len2),dim=1).view(-1, len1, len2)
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, len1),dim=1).view(-1, len2, len1)
        # batch_size x len2 x len1

        sent1_combine = torch.cat(
            (sent1_linear, torch.bmm(prob1, sent2_linear)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat(
            (sent2_linear, torch.bmm(prob2, sent1_linear)), 2)
        # batch_size x len2 x (hidden_size x 2)

        '''sum'''
        g1 = self.mlp_g(sent1_combine.view(-1, 2 * self.embed_size))
        g2 = self.mlp_g(sent2_combine.view(-1, 2 * self.embed_size))
        g1 = g1.view(-1, len1, self.embed_size)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, len2, self.embed_size)
        # batch_size x len2 x hidden_size

        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat((sent1_output, sent2_output), 1)
        # batch_size x (2 * hidden_size)
        h = self.mlp_h(input_combine)
        # batch_size * hidden_size

        h = self.final_linear(h)


        return h
