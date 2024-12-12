# Multiclass Binary Classification of Chest X-Ray Images with MedMNIST  
**Using Vision Transformer, SqueezeNet (Transfer Learning) and Vanilla CNN - Using Loss Curves and GRAD-CAM**

## About the Dataset:
  - The ChestMNIST is based on the NIH-ChestXray14 dataset, comprising 112,120 frontal-view X-Ray images of 30,805 unique patients 14 disease labels, which could be formulized as a multi-label binary-class classification task. MedMNIST resizes all the source images to 1×28×28 for consistency. In this notebook, the images are resized to 3x224x224 to match the SqueezeNet1.1 image dataset size for Transfer Learning.
  - BCELogitLoss is used as it is a multiclass binary classification task.
  - The dataset is divided into Training: 78468, Validation: 11219, Test: 22433
  - Conditions and number of images per condition test set:
      - Class 0: Atelectasis (2,420 images): Partial or complete lung collapse, where lung tissue is unable to inflate properly
      - Class 1: Cardiomegaly (582 images): Enlarged heart
      - Class 2: Effusion (2,754 images): Abnormal fluid accumulation around the lungs seen as a white area on chest X-rays
      - Class 3: Infiltration (3,938 images): Abnormal material present within the lung tissue, often indicating infection, inflammation, or fluid accumulation
      - Class 4: Mass (1,133 images): Abnormal growth or lump in the lung
      - Class 5: Nodule (1,335 images): Small, localized growth or lesion in the lungs
      - Class 6: Pneumonia (242 images): Infection causing inflammation of the air sacs in the lungs
      - Class 7: Pneumothorax (1,089 images): Collapsed lung, where air enters the space between the lung and chest wall
      - Class 8: Consolidation (957 images): Lung tissue becomes more solid-like
      - Class 9: Edema (413 images): Fluid accumulation in lung tissues, causing swelling
      - Class 10: Emphysema (509 images): Chronic lung condition where air sacs are damaged, causing breathing difficulties
      - Class 11: Fibrosis (362 images): Scarring of lung tissue, leading to thickening and stiffening which can impair breathing
      - Class 12: Pleural (734 images): Relates to the pleura, the membrane surrounding the lungs (potentially indicating pleural thickening or effusion)
      - Class 13: Hernia (42 images): Protrusion of an organ or tissue through a weak spot in the surrounding muscle or connective tissue


![X-Ray](https://github.com/user-attachments/assets/25decc68-4b1b-4d91-8f41-29dac9647ce4)


  
## Models Used: 
1. CNN with 3 layers on 28*28 images with 1 channel
2. SqueezeNet - Transfer Learning 224*224 image and 3 channels
3. VisionTransformer Custom Made 28*28 with 1 channel


## Why I chose the models I chose:

![image](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*2mhdctuS-KljkXZB.png)
*Image Source: [Medium](https://medium.com/@avidrishik/squeezenets-architecture-compressed-neural-network-7741d24ca56f)*

- Fire Blocks in Squeeze Net:
  - SqueezeNet is to strike a balance between high accuracy and low complexity great for embedded systems.
 
- Vision Transformer:
  - Vision Transformers is good to use where global dependencies and contextual understanding is important. 


## Loss Curves:

![simple-net](https://github.com/user-attachments/assets/97c5b1f6-621f-4ecf-9d5a-129f43e9e971)

![squeeze-net-tl](https://github.com/user-attachments/assets/159cd7b9-df6c-4e64-bc5d-fcc7f32b8b79)

![homemad-vit](https://github.com/user-attachments/assets/1087d89c-ff21-41ca-a7c2-491aa42fbc2d)


## Accuracy:

| Model             | Validation Accuracy | Test Accuracy | Time of Training | Number of Parameters |
|-------------------|---------------------|---------------|------------------|-----------------------|
| Simple Net        |    95.01%           |    94.59%     |    39 sec/Epoch  |        596,654        |
| Squeeze Net TL    |    94.86%           |    94.74%     |    21 min/Epoch  |727,000 (frozen)+ 7,182|
| Vision Transformer|    94.85%           |    94.74%     |    59 sec/Epoch  |        206,094        |


## GRAD-CAM Images for all the 14 issues: 
- Samples used Image index 5, 6, 8, 13, 24, 42, 62, 91, 108, 243, 388
  
  - Image Index 5: Infiltration AND Nodule
    
  ![ImageIndex5-Class3and5](https://github.com/user-attachments/assets/37115bac-fa69-4f8b-8e21-507853551c74)

  - Image Index 6: Infiltration, Pneumothorax, Consolidation AND Emphysema
    
  ![ImageIndex6-Class3-7-8-10](https://github.com/user-attachments/assets/d5000fff-0eeb-4ec2-b381-326f96d2ee4c)

  - Image Index 13: Cardiomegaly

  ![ImageIndex13-Class1](https://github.com/user-attachments/assets/016cea4f-23eb-4865-8174-ff22adab636f)


  - Image Index 6: Effusion AND Infiltration
    
  ![ImageIndex8-Class2and3](https://github.com/user-attachments/assets/9989fd48-de4d-4261-8898-d4f39fabff33)


  - Image Index 24: Atelectasis AND Emphysema
    
  ![ImageIndex24-Class0-10](https://github.com/user-attachments/assets/05096b0c-699d-43b7-8ac7-69919db038cc)


  - Image Index 42: Atelectasis, Infiltration AND Pneumonia

  ![ImageIndex42-Class0-3-6png](https://github.com/user-attachments/assets/71830691-7e23-44b4-90c5-c0cd5a1927cd)

   - Image Index 62: Effusion, Infiltration, Mass AND Consolidation

  ![ImageIndex62-Class2-3-4-8](https://github.com/user-attachments/assets/e3d1705a-57c0-41cf-97c8-6e9337fc2e7c)

   - Image Index 91: Pleural
  ![ImageIndex91-Class12](https://github.com/user-attachments/assets/a162df38-5471-460b-acb6-4c9641e812c8)


   - Image Index 108: Atelectasis, Infiltration AND Edema
  ![ImageIndex108-Class2-3-9](https://github.com/user-attachments/assets/8eb340bd-46b0-4de2-a26a-ccf51776bd29)


   - Image Index 243: Fibrosis
  ![ImageIndex243-Class11](https://github.com/user-attachments/assets/f48830ce-8682-4439-9ace-d2e4df717af5)


   - Image Index 388: Atelectasis, Fibrosis  AND Hernia
  ![ImageIndex388-Class0-11-13](https://github.com/user-attachments/assets/61183db6-f83f-4bd1-a807-b0afe005ca2c)



## Conclusion:

- Eventhough there are many powerful models out there. It can still be beneficial to look at SimpleCNN as a starting point for a new dataset. It is sometimes the cheapest and most robust model to use like in this case. Feel free to download the models and test it for your own chest-xray dataset. Perhaps a different approach can improve the accuracies further. 



