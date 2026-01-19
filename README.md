# wafer-anomaly-detection

This project implements CS-Flow, a normalizing flow–based method designed for unsupervised industrial anomaly detection.

pip install uv
uv venv
.venv\Scripts\activate
uv init
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install efficientnet_pytorch

uv add torch torchvision
uv add efficientnet_pytorch
uv add FrEIA
uv add PyYaml
uv add tqdm
uv add scikit-learn
uv add scikit-image
uv add torcheval



## References

[1] T. Lei, B. Wang, S. Chen, S. Cao, and N. Zou,  
*Texture-AD: An Anomaly Detection Dataset and Benchmark for Real Algorithm Development*,  
arXiv preprint arXiv:2409.06367, 2024.

[2] M. Rudolph, T. Wehrbein, B. Rosenhahn, and B. Wandt,  
*Fully Convolutional Cross-Scale-Flows for Image-Based Defect Detection*,  
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022.