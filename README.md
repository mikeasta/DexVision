# DexNet: Pokemon Classifier
My first computer vision, object classifying project.

## Workflow
Within this project:
- Custom pokemon dataset was collected and labeled (300 train and 100 test pokemon images).
- Custom torch.utils.data.Dataset class was created for valid image and label load.
- Convolutional neural network with custom architecture (inspired by TinyVGG) was created and trained in 200 epochs.
- Several evaluate and visualization module was written
  
## Results
PokemonClassifierModel state at July 12, 2023: 78.9% test ccuracy in 200 epochs. You can check demo of model predictions in `data/output` folder.


