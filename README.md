## Cloud Coverage Calculator via Sky-Cam Images

#### Project Aim: To find the cloud coverage in percentage from a Skycam image.

#### Domain: Computer Vision | Regression | Image Processing

#### Links: [Data](https://www.allskycam.com/)  |  [Web App](https://huggingface.co/spaces/mykeysid10/gradio-cloud-coverage)  |  [Demo Video](https://www.youtube.com/watch?v=b8qGr6CowWs)  |  [GFG Article](https://www.geeksforgeeks.org/skycam-images-based-cloud-coverage-prediction-via-computer-vision-machine-learning/)  |  [Project Documentation](https://github.com/mykeysid10/EcoTech-Data-Science-GfG-Hackathon-Cloud-Coverage-Calculator/blob/main/Project_Documentation.pdf)

#### Objectives:
1. Cloud Coverage Prediction: To develop a robust model that accurately calculates cloud coverage from skycam images. This model aims to analyze the cloud formations in the provided images and provide a percentage indicating the extent of cloud coverage.
2. Automation: Automate the process of cloud coverage assessment using sky images. This will reduce the need for manual monitoring and provide real-time information on the cloud conditions.

#### Domain Knowledge: 
- Skycam is an automated camera system to periodically record images of the entire sky from dusk until dawn.
- Skycam Image is generated from a Skycam device.
- Low CC: `0% - 33%` | Moderate CC: `33% - 66%` | High CC: `66% - 100%`

<div align="center">
  <img src="https://raw.githubusercontent.com/mykeysid10/EcoTech-Data-Science-GfG-Hackathon-Cloud-Coverage-Calculator/main/Sample_UI_Test_Set/low/20160826164000.raw.jpg" width="30%" alt="Image 1">
  <img src="https://raw.githubusercontent.com/mykeysid10/EcoTech-Data-Science-GfG-Hackathon-Cloud-Coverage-Calculator/main/Sample_UI_Test_Set/moderate/20160304123000.raw.jpg" width="30%" alt="Image 2">
  <img src="https://raw.githubusercontent.com/mykeysid10/EcoTech-Data-Science-GfG-Hackathon-Cloud-Coverage-Calculator/main/Sample_UI_Test_Set/high/20210705150000.raw.jpg" width="30%" alt="Image 3">
</div>

#### Workflow: 

![Workflow](https://raw.githubusercontent.com/mykeysid10/EcoTech-Data-Science-GfG-Hackathon-Cloud-Coverage-Calculator/main/Images/System_Architecture.png)

#### Set Up Steps:

1. Create a python env in your project directory.
2. Install all the depencies using: `pip install -r requirements.txt`
3. Keep `app.py`, `cloud_coverage_pipeline.py`, `catboost_model.sav` & `clip_model.pt` in same directory.
4. Run the `app.py` file and test the application.

#### Demo Video: 

https://github.com/mykeysid10/EcoTech-Data-Science-GfG-Hackathon-Cloud-Coverage-Calculator/assets/70707011/310d84ff-281c-421e-858a-89a986812077

#### Future Scope:
- Accurate weather monitoring is crucial for various applications including agriculture and disaster management. Cloud coverage is a key parameter in weather forecasting and automating its assessment can improve weather predictions.
- Providing real-time information on cloud coverage can benefit industries that rely on weather conditions, such as renewable energy generation, outdoor event planning, and transportation.
- The integration of the cloud coverage model with skycam can serve as an early warning system for impending storms or heavy rains and climatic drifts. This can help in taking preventive measures and ensuring public safety.
