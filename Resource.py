class ResourceManager:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Initialize resource manager state here
        self.device_models = {}
    
    def deploy_model(self, model_name, device):
        if device not in self.device_models:
            self.device_models[device] = []
        self.device_models[device].append(model_name)
        print(f"Model '{model_name}' deployed to device '{device}'")
    
    def get_models_on_device(self, device):
        return self.device_models.get(device, [])
    
# Usage
if __name__ == "__main__":
    # Creating resource manager instance
    rm = ResourceManager()
    
    # Deploying models to devices
    rm.deploy_model("Model 1", "Device 1")
    rm.deploy_model("Model 2", "Device 2")
    rm.deploy_model("Model 3", "Device 3")
    
    # Retrieving models deployed on devices
    print("Models on Device 1:", rm.get_models_on_device("Device 1"))
    print("Models on Device 2:", rm.get_models_on_device("Device 2"))
    print("Models on Device 3:", rm.get_models_on_device("Device 3"))