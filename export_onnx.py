from ultralytics import YOLO

# 1. Cargar el modelo entrenado
model = YOLO("models/best.pt")

# 2. Exportar a formato ONNX
model.export(
    format="onnx",
    dynamic=True,      # Permite tamaños de batch e imagen dinámicos
    simplify=True,     # Optimiza el grafo del modelo eliminando nodos redundantes
    opset=12           # Versión de operadores compatible con la mayoría de runtimes
)