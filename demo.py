import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn
import models.crnn_lite as crnn_lite

model_path = 'expr_lite/netCRNN_199.pth'
img_path = 'data/origin_data_val/100.png'
alphabet = '0123456789'

# model = crnn_lite.CRNN(32, 1, 11, 256)
model = crnn_lite.CRnn(32, 1, 11, 256)  #第二个参数1代表input的channel，1是灰度图
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
