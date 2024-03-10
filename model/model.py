# model.py
# We are going to use onnx based model
import onnx
import numpy as np

# TVM
# import tvm
# from tvm import relay
# from tvm import te
# from tvm import rpc
# from tvm.contrib import utils
# from tvm.relay import testing
# from tvm.contrib import graph_executor

# Load Pillow for loading example input
# from PIL import Image

# Import pandas where all the configuration is saved in platforms.csv and pretrained_models.csv
import pandas as pd

# Assumptions
# 1. Every ONNX based models must be stored in database
# 2. User will select available model from the database
# 3. Custom compilation (as defined in platform)
# 4. Load compiled model

# For now we are going to compile using TVM
class Model:
    def __init__(self, model_name, platform, compiled_model=None):
        self.model_name = model_name
        
        self.platform = platform
        # self.example_np_input = example_np_input
        self.compiled_model = compiled_model
        # self.input_shape = None
        # self.output_shape = None
        
        # self.output = model.graph.output
        # self.input_all = model.graph.input
        
        # Load database file of stored pretrained models and platforms
        df_pretrained_models = pd.read_csv("pretrained_models/pretrained_models.csv") # Reading stored models
        df_platforms = pd.read_csv("platforms/platforms.csv") # Reading stored platforms

        self.pretrained_model_path = df_pretrained_models[df_pretrained_models["Model"] == self.model_name]['Model_Path'].tolist()[0]
        self.onnx_model = onnx.load(self.pretrained_model_path)
        
        self.pretrained_model_input_name = df_pretrained_models[df_pretrained_models["Model"] == self.model_name]['Input_Name'].tolist()[0]
        self.pretrained_model_output_name = df_pretrained_models[df_pretrained_models["Model"] == self.model_name]['Output_Name'].tolist()[0]
        
        self.platform_parameters = df_platforms[df_platforms["Platform"] == self.platform]['TVMParameters'].tolist()[0]
        self.platform_device = df_platforms[df_platforms["Platform"] == self.platform]['Device'].tolist()[0]
        
        shape_dict = {}
        for input in self.onnx_model.graph.input:
            shape = [dim.dim_value if dim.dim_value is not None else 1 for dim in input.type.tensor_type.shape.dim]
            shape_dict[input.name] = shape
        self.input_shape = shape_dict[self.pretrained_model_input_name] 
        
        # Create TVM context
        # self.ctx = tvm.device(str(self.platform_parameters)) # Create context  
        # print(self.input_shape)
        
    def compile(self, compile_to_path):
        # Your implementation for the compile method
        # Preparing IR
        shape_dict = {inputname_models[model_name]: self.example_np_input.shape}
        mod, params = relay.frontend.from_onnx(self.model_path, shape_dict)
        
        # Compiling
        opt_level = 3
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(mod, target_str, params=params)
            
        if ".tar" in compile_to_path:
            lib.export_library(compile_to_path)
        else:
            lib.export_library(f"{compile_to_path}.tar")

    def load(self):
        if self.executable is None:
            raise RuntimeError("Model has not been compiled yet.")
        
        ####### Ignore this, since it has been defined in constructor ############
        # ctx = tvm.device(self.platform_device)
        # ctx = tvm.device(str(self.platform_parameters), 0)
        # ctx = tvm.device(str(self.platform_parameters))
        ##########################################################################
        
        self.module = graph_runtime.GraphModule(self.compiled_model["default"](ctx))
        
    def predict(self, input):
        if self.module is None:
            raise RuntimeError("Model has not been loaded yet.")
        
        # Check if the number of inputs matches the expected number
        if input.shape != self.input_shape:
            raise ValueError(f"Expected {self.input_shape} inputs, but got {input.shape}.")
        
        # Set input data
        # for i, (name, shape) in enumerate(self.input_shape.items()):
        #     self.module.set_input(name, tvm.nd.array(inputs[i].reshape(shape).astype('float32')))
        
        input_data = tvm.nd.array(input.astype('float32'), self.ctx)
        self.module.set_input(self.pretrained_model_input_name, input_data)
        
        # Run inference
        self.module.run()
        
        # Get output
        output = self.module.get_output(0).asnumpy()
        preds = np.squeeze(out_deploy)
        probs = softmax(preds)
        return preds, probs