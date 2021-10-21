#load model
import models.crnn_lite as crnn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

crnn = crnn.CRnn(32, 1, 11, 256) #Lite version
crnn.apply(weights_init)
state_dict = crnn.state_dict()

pre_t_lite = torch.load("data/crnn_lite_lstm_dw.pth")

pass