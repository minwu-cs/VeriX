import argparse
import tf2onnx
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--output_path', type=str, default='networks/')
args = parser.parse_args()

model_path = args.model_path
out_path = args.output_path
split = model_path.rsplit(model_path, 1)
model_name = split[1] if len(split) > 1 else model_path

if model_name[-3:] == '.h5':
    model = load_model()
    model_proto, _ = tf2onnx.convert.from_keras(model, output_path=out_path + model_name[-3:] + '.onnx')
