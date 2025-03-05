import onnx

AGE_MODEL_PATH = "models/face_analysis/age_googlenet.onnx"
GENDER_MODEL_PATH = "models/face_analysis/gender_googlenet.onnx"

def check_onnx_inputs(model_path):
    """Prints input names of an ONNX model."""
    model = onnx.load(model_path)
    input_all = [node.name for node in model.graph.input]
    print(f"Inputs for {model_path}: {input_all}")

check_onnx_inputs(AGE_MODEL_PATH)
check_onnx_inputs(GENDER_MODEL_PATH)
