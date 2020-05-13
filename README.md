## Project on Self-Supervised Learning for Prof. Yann LeCun's Deep Learning Course at NYU, Spring 2020
 
1. Model architecture and training files:

   - [Denoising Auto-encoder](https://github.com/MrinalJain17/ssl-project/blob/master/denoising_autoencoder.py)
   - [Road Map Construction](https://github.com/MrinalJain17/ssl-project/blob/master/road_map_construction.py)

   ```
   python road_map_construction.py --EPOCHS 50 --BATCH_SIZE 128
   ```

2. Sample notebook showing the generated roadmaps on validation data: [Link](https://github.com/MrinalJain17/ssl-project/blob/master/road_map_construction.ipynb)

3. The directory `saved_models` contains the trained model checkpoints.

4. The code for object detection is inside the directory `object_detection`.

### Requirements

1. PyTorch
2. Torchvision
3. [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - for simplifying the training, validation and testing interface.
