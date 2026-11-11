from pathlib import Path
import tensorflow as tf

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "edge_ai"
    / "vision"
    / "models"
    / "best_eye_model_v4_smartcap.keras"
)

print("Loading SmartCap V4 model...")

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)