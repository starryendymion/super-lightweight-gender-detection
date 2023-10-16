# Super-lightweight-gender-detection

<h3>Information</h3>
→ This is a highly optimized classification model that runs on a Tflite interpreter library.Full tensorflow is not needed! <br>
→ Principal Component Analysis (PCA) is used to reduce the feature dimensions from 40 to 20 while maintaining a 95% representation of the data.<br>
→ Quantization-aware training and post-training quantization are applied to the model to optimize it further.<br>
→ Final size of the .tflite model : 2.9 KB<br>
→ This model runs as a streamlit.io app.<br>

<h2>Requirements</h2>
→ Python3 | streamlit | scikit-learn | tflite_interpreter | Numpy+Pandas <br>

<h2>How to run</h2>
→ Make sure that transform modules, model files, and preprocessing files are in the right place.<br>
→ After making sure that required libraries are installed, execute web_app.py in terminal
