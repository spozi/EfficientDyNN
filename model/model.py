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

# Import pandas where all the configuration is saved in platforms and pretrained_models
import pandas as pd

# For now we are going to compile using TVM
class Model:
    def __init__(self, model, platform, example_np_input=None):
        self.model = model
        self.platform = platform
        self.example_np_input = example_np_input
        
        # Load database file of stored pretrained models and platforms
        df_pretrained_models = pd.read_csv("pretrained_models/pretrained_models.csv") # Reading stored models
        df_platforms = pd.read_csv("platforms/platforms.csv") # Reading stored platforms

        self.pretrained_model_path = df_pretrained_models[df_pretrained_models["Model"] == self.model]['Model_Path'].tolist()[0]
        self.pretrained_model_input_name = df_pretrained_models[df_pretrained_models["Model"] == self.model]['Input_Name'].tolist()[0]
        self.pretrained_model_output_name = df_pretrained_models[df_pretrained_models["Model"] == self.model]['Output_Name'].tolist()[0]
        self.platform_parameters = df_platforms[df_platforms["Platform"] == self.platform]['TVMParameters'].tolist()[0]
        
        # print(self.platform_parameters, self.pretrained_model_path, self.pretrained_model_input_name, self.pretrained_model_output_name)
    
    
    
    def compile(self, compile_to_path):
        # Your implementation for the compile method
        print("Compiling")
        # Preparing IR
        shape_dict = {inputname_models[model_name]: self.example_np_input.shape}
        mod, params = relay.frontend.from_onnx(self.model_path, shape_dict)
        pass

    def predict(self, data):
        # Your implementation for the predict method
        print("Predicting")
        pass