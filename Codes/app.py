# Importing Libraries
import gradio as gr
from gradio.components import Image, Textbox
from cloud_coverage_pipeline import predict_cloud_coverage, initialize_models

# Initialize the CLIP model and CatBoost model only once
cbt_model, clip_model = initialize_models()

# Method to call pipeline.py to calculate cloud coverage
def predict(image):
    if image is None:
        return "Please Upload a valid sky image!"
    pred_cloud_coverage = predict_cloud_coverage(image, clip_model, cbt_model)
    if pred_cloud_coverage <= 33.0:
        s = "There is Low Cloud Coverage!   Predicted Opaque Cloud Coverage: {}%".format(pred_cloud_coverage)
    elif pred_cloud_coverage > 33.0 and pred_cloud_coverage <= 66.0:
        s = "There is Moderate Cloud Coverage!   Predicted Opaque Cloud Coverage: {}%".format(pred_cloud_coverage)
    else:
        s = "There is High Cloud Coverage!   Predicted Opaque Cloud Coverage: {}%".format(pred_cloud_coverage)
    return s

# Create the Gradio app
iface = gr.Interface(
    fn = predict,
    inputs = [Image(label = "Upload a Sky Cam image")],
    outputs = [Textbox(label = "Prediction")],
    title = "GFG EcoTech Hackathon: Cloud Coverage Calculator From a Sky Cam Image",
    description = 'Upload only a skycam image and get the opaque cloud coverage in %     |     (Low: 0-33 | Moderate: 33-66 | High: 66-100)     |     <a href="https://drive.google.com/drive/folders/1r8mTWEG4XEBZDg0TNyXTYkGzZVixXvcj?usp=drive_link">Find Sample Testing Images Here!</a>',
)

# Run the Gradio app
iface.launch(debug = True)