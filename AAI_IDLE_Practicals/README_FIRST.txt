ADVANCED ARTIFICIAL INTELLIGENCE PRACTICALS — PYTHON IDLE PACK

HOW TO USE
1. Extract this ZIP to a folder.
2. Open IDLE using the same Python installation where the required packages are installed.
3. In IDLE choose File > Open and select ONE practical .py file.
4. Press F5 (Run > Run Module).
5. Install only the packages required by that practical, using Command Prompt (CMD), not the IDLE Shell.

IMPORTANT: TensorFlow was previously installed for this Python executable:
C:\Users\AKANKSHA\AppData\Local\Programs\Python\Python310\python.exe

To open the matching IDLE, run this in CMD:
C:\Users\AKANKSHA\AppData\Local\Programs\Python\Python310\python.exe -m idlelib

INSTALL PACKAGES
Run these commands in CMD (not in the IDLE Shell). Use the Python executable above.

Practical 1:
python -m pip install tensorflow matplotlib

Practical 2:
python -m pip install tensorflow matplotlib

Practical 3:
python -m pip install transformers torch

Practical 4:
python -m pip install tensorflow numpy scikit-learn

Practical 5:
python -m pip install ultralytics opencv-python matplotlib

Practical 6:
python -m pip install numpy

Practical 7:
python -m pip install tensorflow pandas matplotlib scikit-learn

Practical 8:
python -m pip install scikit-learn pandas

Practical 9A:
python -m pip install optuna scikit-learn

Practical 9B (optional; TPOT can be difficult to install on some Python versions):
python -m pip install tpot scikit-learn

Practical 10A:
python -m pip install scikit-learn joblib

Practical 10B:
python -m pip install flask scikit-learn joblib

NOTES
- Run each practical separately. Do not paste all practicals into one IDLE file.
- Some practicals download datasets or pretrained models on the first run and need internet.
- Practical 1 and 2 may take a few minutes. Start with fewer epochs if your computer is slow.
- Practical 3 downloads GPT-2 and may use substantial disk space/memory.
- Practical 5 downloads the YOLOv8 model and sample image.
- Practical 8 uses scikit-learn GridSearchCV as a simpler, more IDLE-friendly model-selection pipeline than TPOT.
- Practical 9 has two files: 9A (Optuna Bayesian optimization) and 9B (TPOT evolutionary search).
- Practical 10 is split into two files: first train/save the model with 10A, then run 10B to serve predictions locally. This is a small local deployment demonstration, not a cloud deployment.
- Practical 4 uses a tiny illustrative sample dataset, so its predictions are for demonstration only.
