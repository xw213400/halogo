Step 1: cpp/bin/xrecordtf generate games (data/train data/estimate)
Step 2: py/xtrain.py generate pytorch module (data/goai.pth)
Step 3: py/xgenonnx.py convert pytorch to onnx (data/goai.onnx)
Step 4: py/xonnx2tf.py convert onnx to tensorflow (data/goai.pb)
Step 5: mv data data_xxx && mkdir data && cp data_xxx/goai.pb data
Step 6: goto step 1