

import torch
import torch.onnx
import torch._utils
from collections import OrderedDict

import onnx

from models.crnn_lite import CRnn
import dataset
import onnxruntime
from PIL import Image
import utils
from torch.autograd import Variable

alphabet = '0123456789'

def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def lstm_op_adapter(state_dict):
    ret = {}
    for key, value in state_dict.items():
        if not (key.startswith('rnn') and ('fw' in key or 'bw' in key)):
            ret[key] = value
            continue
        param = state_dict[key].data.split(256)
        ret[key] = torch.cat((param[0], param[2], param[1], param[3]), 0)
    return ret


def convert(pth_file_path, onnx_path):
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    #checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model = CRnn(32, 1, 11, 256) #imgH, nc, nclass, nh

    #model.load_state_dict(lstm_op_adapter(checkpoint))
    model.load_state_dict(checkpoint)

    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    batch_size = 1
    dummy_input = torch.randn(batch_size, 1, 32, 100)
    dynamic_axes = {
        "actual_input_1": {0: "batch_size"}, "output1": {1: "batch_size"}
    }
    # dynamic_axes 将actual_input_1的第0个值改为-1,本来这个位置是batch_size,
    #              将actual_input_1的第1个值改为-1,本来这个位置是batch_size,

    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11)


def test_onnx(path):
    session = onnxruntime.InferenceSession(path)

    session.get_modelmeta()
    first_input_name = session.get_inputs()[0].name
    first_output_name = session.get_outputs()[0].name

    input = get_and_preprocessImg()

    results = session.run([], {first_input_name: input.numpy()})
    pred = results[0].argmax(2)#(26,1)
    pred_tensor = torch.from_numpy(pred)
    preds_size_tensor = Variable(torch.IntTensor([pred.size]))  # 转为tensor

    converter = utils.strLabelConverter(alphabet)
    raw_pred = converter.decode(pred_tensor, preds_size_tensor, raw=True)
    sim_pred = converter.decode(pred_tensor, preds_size_tensor, raw=False)
    print('{:20s} => {:3s}'.format(raw_pred, sim_pred))

def get_and_preprocessImg():
    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open("data/origin_data_val/101.png")
    image = image.convert('L')
    image = transformer(image)
    return image.view(1, *image.size())


def simply_onnx():
    from onnxsim import simplify
    in_path = "onnx/crnn_lite.onnx"
    output_path = "onnx/simply/crnn_lite_simply.onnx"
    onnx_model = onnx.load(in_path)  # load onnx model
    model_simp, check = simplify(onnx_model, dynamic_input_shape=True, input_shapes={"actual_input_1": [1, 1, 32, 100]})
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    print('finished exporting onnx')


if __name__ == "__main__":
    ###1
    # pth_file_path = "expr_lite/netCRNN_199.pth"
    # onnx_path = "onnx/crnn_lite.onnx"
    # convert(pth_file_path, onnx_path)

    ###2
    #test_onnx("onnx/crnn_lite.onnx")

    ###3
    #还需要使用pip install onnx-simplifier简化
    simply_onnx()

    pass