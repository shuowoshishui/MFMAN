# MFMAN
# Haze image recognition through multi-scale feature and multi-adversarial<br>
### This is the code of MFMAN<br>
proposed a new diversity real haze dataset<br>
dataset name Real-haze12,Real-clear12,Resources in Baidu Cloud，link：https://pan.baidu.com/s/1gZrmJsNhiXOWvKwb8pOzuQ :`dnxd`<br>
### Prerequisites:<br>
    Python3
    PyTorch == 1.0.0 (with suitable CUDA and CuDNN version)
    Numpy
    argparse
    PIL
    tqdm
## Baseline
·Deep residual learning(ResNet50)<br>
·Domain Adaptive Neural Network(DANN)<br>
·Deep Domain Confusion (DDC)<br>
·Deep CORAL: Correlation Alignment for Deep Domain Adaptation (D-CORAL)<br>
·Domain-Adversarial Training of Neural Networks (DANN-A) <br>
the parametes is in the baseline\options.py,then run the train.py
## MFMAN
fix the parameters in MFMAN\train.py,then run the train.py
