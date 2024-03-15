# model.py
# We are going to use onnx based model
import onnx
import numpy as np
import timeit

# TVM
import tvm
from tvm import relay, autotvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils
from tvm.relay import testing
from tvm.contrib import graph_executor, graph_runtime
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner

# Load Pillow for loading example input
# from PIL import Image

# Import pandas where all the configuration is saved in platforms.csv and pretrained_models.csv
import pandas as pd

# Import softmax from scipy
from scipy.special import softmax, log_softmax

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
        self.compiled_model = compiled_model
        
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
        self.ctx = tvm.device(str(self.platform_parameters)) # Create context  
        # print(self.input_shape)
        
    def compile(self, compile_to_path, performance_tuning=False):
        # Preparing IR
        shape_dict = {self.pretrained_model_input_name : self.input_shape}
        mod, params = relay.frontend.from_onnx(self.onnx_model, shape_dict)
        
        # Compiling
        target = tvm.target.Target(str(self.platform_parameters))
        opt_level = 3
        lib = 0  # Declare
        if performance_tuning is True:           
            # Start tuning
            # Create TVM Runner
            number = 10
            repeat = 1
            min_repeat_ms = 0  # since we're tuning on a CPU, this can be set to 0
            timeout = 10  # in seconds
            runner = autotvm.LocalRunner(
                number=number,   #Number means number of variations that will be tested
                repeat=repeat,
                timeout=timeout,
                min_repeat_ms=min_repeat_ms,
                enable_cpu_cache_flush=True,
            )
            
            tuning_option = {
                "tuner": "xgb",
                "trials": 20,
                "early_stopping": 100,
                "measure_option": autotvm.measure_option(
                    builder=autotvm.LocalBuilder(build_func="default"), runner=runner
                ),
                "tuning_records": f"{compile_to_path}-autotuning.json",
            }

            # begin by extracting the tasks from the onnx model
            tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

            for i, task in enumerate(tasks):
                prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

                # choose tuner
                tuner = "xgb"

                # create tuner
                if tuner == "xgb":
                    tuner_obj = XGBTuner(task, loss_type="reg")
                elif tuner == "xgb_knob":
                    tuner_obj = XGBTuner(task, loss_type="reg", feature_type="knob")
                elif tuner == "xgb_itervar":
                    tuner_obj = XGBTuner(task, loss_type="reg", feature_type="itervar")
                elif tuner == "xgb_curve":
                    tuner_obj = XGBTuner(task, loss_type="reg", feature_type="curve")
                elif tuner == "xgb_rank":
                    tuner_obj = XGBTuner(task, loss_type="rank")
                elif tuner == "xgb_rank_knob":
                    tuner_obj = XGBTuner(task, loss_type="rank", feature_type="knob")
                elif tuner == "xgb_rank_itervar":
                    tuner_obj = XGBTuner(task, loss_type="rank", feature_type="itervar")
                elif tuner == "xgb_rank_curve":
                    tuner_obj = XGBTuner(task, loss_type="rank", feature_type="curve")
                elif tuner == "xgb_rank_binary":
                    tuner_obj = XGBTuner(task, loss_type="rank-binary")
                elif tuner == "xgb_rank_binary_knob":
                    tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="knob")
                elif tuner == "xgb_rank_binary_itervar":
                    tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="itervar")
                elif tuner == "xgb_rank_binary_curve":
                    tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="curve")
                # elif tuner == "ga":
                #     tuner_obj = GATuner(task, pop_size=50)
                # elif tuner == "random":
                #     tuner_obj = RandomTuner(task)
                # elif tuner == "gridsearch":
                #     tuner_obj = GridSearchTuner(task)
                else:
                    raise ValueError("Invalid tuner: " + tuner)
                
                tuner_obj.tune(
                    n_trial=min(tuning_option["trials"], len(task.config_space)),
                    early_stopping=tuning_option["early_stopping"],
                    measure_option=tuning_option["measure_option"],
                    callbacks=[
                        autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                        autotvm.callback.log_to_file(tuning_option["tuning_records"]),
                    ],
                )
                with autotvm.apply_history_best(tuning_option["tuning_records"]):
                    with tvm.transform.PassContext(opt_level=3, config={}):
                        lib = relay.build(mod, target=target, params=params)

                # dev = tvm.device(str(target), 0)
                # module = graph_executor.GraphModule(lib["default"](dev))
        else:
            with tvm.transform.PassContext(opt_level=opt_level):
                lib = relay.build(mod, target, params=params)
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
        self.module = graph_executor.GraphModule(self.compiled_model["default"](self.ctx)) 
        
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
        preds = np.squeeze(output)
        probs = softmax(preds)
        return preds, probs