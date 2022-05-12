Мое решение тестового задания на позицию Middle Outstaff Researcher. 

#### Структура:
```
.
├── cfg                   
│   ├── 64_200.json    
│   └── example.json
├── chkp                  
│   ├── 64_200  
│   ├── 64_200_int2
│   ├── 64_200_int4
│   └── 64_200_int8  
├── src                    
│   ├── model.py                 
│   ├── qmodel.py                
│   ├── train.py                
│   ├── utils.py  
│   └── __init__.py
├── PTQ.ipynb
├── train_resnet.py
├── requirements.txt
└── README.md
```
Реализованно
[Vladimir Kondratyev](https://github.com/VldKnd)

---
**Шаг 1.** Ознакомиться с понятием Quantization.  

На данном шаге предлагается разобраться с понятием Quantization и со стандартными техниками, как dynamic, static, post training и quantization aware quantization.

**Шаг 2.** Скачать датасет CIFAR10.

**Шаг 3.** Реализовать архитектуру ResNet20.

**Шаг 4.** Обучить ResNet20.  

**Шаг 5.** Применить готовые решения для post training quantization (далее PTQ).  

> Квантовать к 16 и 8 битам. Квантовать уже обученную модель, которая была получена на шаге 4.

**Шаг 6.** Реализовать PTQ.

> Квантовать к 16, 8, 4 и 2 битам. Квантовать уже обученную модель, которая была получена на шаге 4.

**Шаг 7.** Сравнение результатов

---
