# DexNet: Pokemon Classifier
My first computer vision object classifying project.

## How to use
1. Clone repository on your machine with:
```bash
git clone https://github.com/mikeasta/DexVision/
```
2. Copy some pokemon images into `DexVision/data/input` (remove all images in this dir before copying).
3. Run `DexVision/dexnet/evaluate.py` script with:
```bash
python3 DexVision/dexnet/evaluate.py
```
4. Check `DexVision/data/output` directory.

## Workflow
Within this project:
- Custom pokemon dataset was collected and labeled (300 train and 100 test pokemon images).
- Custom torch.utils.data.Dataset class was created for valid image and label load.
- Convolutional neural network with custom architecture (inspired by TinyVGG) was created and trained with 200 epochs.
- Several evaluate and visualization modules was written.
  
## Results
PokemonClassifierModel state at July 12, 2023: 78.9% test ccuracy in 200 epochs. You can check demo of model predictions in `data/output` folder.
