This is my implementation of variation of quantization of neural network called static post training quantization. The sources i found useful while reading about it:


#### Structure:
```
.
├── cfg                   
│   ├── 64_200.json     ### Configuration file with neural network and DataLoader config.
│   └── example.json
├── chkp                ### Saved model weights.  
│   ├── 64_200          ### Weights for not quantized network ResNet20, the accuracy on CIFAR10 test 92%.
│   ├── 64_200_int2     ### Quantized to int2.
│   ├── 64_200_int4     ### Quantized to int2.
│   └── 64_200_int8     ### Quantized to int2.
├── src
│   ├── model.py        ### Custom implementation of ResNet blocks and architecture.            
│   ├── qmodel.py       ### Quantized blocks.       
│   ├── train.py        ### Function for training and validation of neural network.
│   ├── utils.py        ### Utilities.
│   └── __init__.py
├── PTQ.ipynb           ### Short notebook with quantization of ResNet20 to int2, int4, int8 and int8 with merged Conv2d and BatchNorm.
│
├── train_resnet.py
├── requirements.txt
└── README.md
```
To download CIFAR10 it is possible to put ```download:true```in the config file. Then while running ```train_resnet.py```will download it to the folder ```\data```. ```train_resnet```can be used to train ResNet20. I have left the possibility to change the hypersettings, such as batch size and number of training epoches. All other parameters are fixed, you can check them out in train_resnet.py. I have used the [architecture](https://www.researchgate.net/figure/ResNet-20-architecture_fig3_351046093) with batch normalization after the convolutions.

Example of configuration:
```json
{
    "_comment_json": "This field is used for comments only",
    
    "cfg":{
        "_comment_chekpoint_path": "The path to save models",
        "checkpoint_path":"./chkp/64_200"
    },

    "cfg_CIFAR":{
        "_comment_root": "Data Folder CIFAR10",
        "root":"./data",
        "_comment_download": "Wether to download the files or not",
        "download":true
    },
    
    "_comment_cfg_dataloader_train":"Configuration for trainloader",
    "cfg_dataloader_train":{
        "batch_size":64,
        "shuffle":true,
        "num_workers":2,
        "pin_memory":true
    },

    "_comment_cfg_dataloader_test":"Configuration for testloader",
    "cfg_dataloader_test":{
        "batch_size":1024,
        "shuffle":false,
        "num_workers":2,
        "pin_memory":true
    },

    "_comment_cfg_train":"Configuration for training cycle",
    "cfg_train":{
        "n_epoches":200
    }
}
```
Usage example of ```train_resnet.py```
```
python train_resnet.py cfg/64_200.json
```

**Results** | Original | PyTorch int2 | PyTorch int4 | PyTorch int8 | Fused PyTorch int8 | My Model int8
------ | ------ | ------ | ------ | ------ | ------ | ------ 
Accuracy | 92.1% | 11.1% | 29.6% | 88.1% | 91.3% | 91.4% 
Inference Time | 39.5s ± 0.8s | 22.9s ± 0.3s | 22.9s ± 0.3s | 22.9s ± 0.3s | 15.6s ± 0.1s | 16.8s ± 0.3s 
Size | 1.596 MB | 0.485 MB | 0.485 MB | 0.485 MB | 0.430 MB | 0.425 MB  


[Vladimir Kondratyev](https://github.com/VldKnd)
