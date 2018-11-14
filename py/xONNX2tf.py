
import onnx
from onnx_tf.backend import prepare

model = onnx.load('../data/goai.onnx')
tf_rep = prepare(model)

print('inputs:', tf_rep.inputs)
print('outputs:', tf_rep.outputs)
print('tensor_dict:')
print(tf_rep.tensor_dict)

tf_rep.export_graph('../data/goai.pb')
