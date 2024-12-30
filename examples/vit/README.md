Applying Monarch-structured attention to base ViT on ImageNet, we can reduce FLOPs in the **attention operation** by roughly 60% with little to no loss in accuracy and without any additional training. 

To test on a few images, run
```
python vit/test_monarch.py
```
from the examples directory. Expected output (on CPU):
```
Softmax attention time: 0.21s
Monarch attention time: 0.56s

Softmax: ['coucal', 'magpie', 'kite', 'bee eater', 'red wolf, maned wolf, Canis rufus, Canis niger'] | Monarch: ['coucal', 'bee eater', 'jacamar', 'bulbul', 'agama'] | True: coucal

Softmax: ['Mexican hairless', 'Italian greyhound', 'Doberman, Doberman pinscher', 'Great Dane', 'Weimaraner'] | Monarch: ['Mexican hairless', 'Italian greyhound', 'Weimaraner', 'Great Dane', 'whippet'] | True: Italian greyhound

Softmax: ['valley, vale', 'alp', 'lakeside, lakeshore', 'volcano', 'sandbar, sand bar'] | Monarch: ['valley, vale', 'sandbar, sand bar', 'lakeside, lakeshore', 'alp', 'dam, dike, dyke'] | True: volcano

Softmax: ['Welsh springer spaniel', 'Sussex spaniel', 'English springer, English springer spaniel', 'clumber, clumber spaniel', 'Brittany spaniel'] | Monarch: ['Welsh springer spaniel', 'Blenheim spaniel', 'Brittany spaniel', 'papillon', 'cocker spaniel, English cocker spaniel, cocker'] | True: Welsh springer spaniel

Softmax: ['chickadee', 'junco, snowbird', 'jay', 'water ouzel, dipper', 'house finch, linnet, Carpodacus mexicanus'] | Monarch: ['chickadee', 'junco, snowbird', 'jay', 'water ouzel, dipper', 'bulbul'] | True: chickadee

Monarch to Softmax FLOP ratio: 1.18e+08/2.98e+08 (39.48%)
```
