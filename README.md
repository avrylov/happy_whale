# happy_whale

[happy whale kaggle competition](https://www.kaggle.com/competitions/happy-whale-and-dolphin)

## two steps approach

### 1. Cropping images
./data/crop/...

![alt text](readme_images/img.png)
![alt text](readme_images/img_1.png)

### 2. Identification whale model
./model/train.py

**check for pairs**
![alt text](readme_images/pair_0.png)
![alt text](readme_images/pair_1.png)

confusion matrix: 
![alt text](readme_images/cm.png)

best result - tf exp 1.1.5 epoch=11-val_loss=0.6487-val_acc=0.6381.ckpt

p.s. project is under construction ;)