# Earthquake Prediction System Using Deep Learning & Email Alerts
This project predicts earthquake Magnitude and Depth using a deep-learning model (Keras).<br>
It also visualizes earthquake locations globally and sends automatic email alerts with prediction results.

# Features
🔄 Converts Date + Time into UNIX timestamps<br>
   Cleans and preprocesses geophysical data<br>
🗺️ Plots earthquake locations using Basemap<br>
   Deep-learning model for:<br>
 •Magnitude prediction<br>
 •Depth prediction<br>
🔍 Hyperparameter tuning with cross-validation<br>
📧 Auto email alerts with prediction results<br>
📉 Scaled outputs using StandardScaler<br>

# 📂 Project Structure
```md

Earthquake-Prediction-System/
├── dataset.csv
├── main.py # ML model + plotting + email alerts
├── requirements.txt
├── README.md
└── images/
└── global_map.png
```
# Technologies Used

► Python 3<br>
► NumPy, Pandas<br>
► Matplotlib, Basemap<br>
► Scikit-learn<br>
► TensorFlow + Keras<br>
►SMTP for Email Alerts<br>

# Installation (Run Locally)
1.)Clone the repository
```md
git clone https://github.com/ShriyaRao16/Earthquake-Prediction.git
cd Earthquake-Prediction
```
2.) Create a virtual environment
```md
python -m venv env
source env/bin/activate          # Windows: env\Scripts\activate
```
3.)Install dependencies
```md
pip install -r requirements.txt
```
If Basemap fails:
```md
pip install https://github.com/matplotlib/basemap/archive/master.zip
```
▶️ Run the Project
```md
python main.py
```
For Email Alert Setup 
Inside main.py
```python
sender_email = "sendersmail@gmail.com"
sender_password = "abcd 1234"
```
Use Gmail App Password:<br>
1.)Turn ON 2-Step Verification<br>
2.)Open App Passwords<br>
3.)Generate password<br>
4.)Replace it in the script<br>

# Model Architecture 
```scss
Input (Timestamp, Latitude, Longitude)
   ↓
Dense(16, relu)
Dense(16, relu)
Dense(2) → (Magnitude, Depth)
```
► Loss: MSE<br>
► Optimizer: Adam<br>
► Metric: MAE<br>

# requirements.txt
```nginx
numpy
pandas
matplotlib
basemap
scikit-learn
tensorflow
keras
scikeras
```

# 📝 End Notes

Thank you for checking out this project! <br>
Feel free to star ⭐ the repo or contribute!




