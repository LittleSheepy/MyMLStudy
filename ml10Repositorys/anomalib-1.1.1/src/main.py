# Import the required modules
from anomalib.data import MVTec
from anomalib.models import Padim
from anomalib.engine import Engine

# Initialize the datamodule, model and engine
datamodule = MVTec()
model = Padim(pre_trained=False)
engine = Engine()

# Train the model
if __name__ == '__main__':
    engine.fit(datamodule=datamodule, model=model)