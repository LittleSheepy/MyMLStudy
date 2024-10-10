# Import the required modules
from anomalib.data import MVTec
from anomalib.models import Padim, get_model
from anomalib.engine import Engine
from anomalib.deploy import ExportType
# Initialize the datamodule, model and engine
datamodule = MVTec()
# model = get_model("Padim", pre_trained=False)
model = Padim(pre_trained=False)
engine = Engine()

# Train the model
if __name__ == '__main__':
    # engine.fit(datamodule=datamodule, model=model)
    engine.export(model=model, export_type=ExportType.ONNX, input_size=(256, 256),
                  ckpt_path = r"D:\00sheepy\01code\01alg_code\AI_UI\AnoUI\trunk\Project_Ano\ANO\models\Padim\MVTec\latest\weights\lightning/model.ckpt"
                  )
    # engine.export(model=model, export_type=ExportType.OPENVINO)
    ckpt_path = r"D:\00sheepy\00MyMLStudy\ml10Repositorys\anomalib-1.1.1\src\results\Padim\MVTec\bottle\latest\weights\lightning/model.ckpt"
    # predictions = engine.predict(datamodule=datamodule, model=model, ckpt_path=ckpt_path)
