from model import Model

model = Model(model="squeezenet1.1-7", platform="TuTJetsonCPU")
# model.compile()

# from Resource import ResourceManager

# # Creating resource manager instance
# rm = ResourceManager()

# # Deploying models to devices
# rm.deploy_model("Model 1", "Device 1") # Configuration file from models and devices
# rm.deploy_model("Model 2", "Device 2")
# rm.deploy_model("Model 3", "Device 3")

# # Retrieving models deployed on devices
# print("Models on Device 1:", rm.get_models_on_device("Device 1"))
# print("Models on Device 2:", rm.get_models_on_device("Device 2"))
# print("Models on Device 3:", rm.get_models_on_device("Device 3"))