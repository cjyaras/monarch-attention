Applying Monarch-structured attention to base ViT on ImageNet, we can reduce FLOPs in the **attention operation** by roughly 60% with little to no loss in accuracy and without any additional training. Even for relatively short sequences, we get a roughly 18% speed-up in inference.

To test on a few images, run
```
python vit/monarch_vit_example.py
```
from the examples directory. Expected output (on A100):
```
Softmax attention time: 18.42ms
Monarch attention time: 15.68ms

Softmax: ['coucal', 'magpie', 'kite', 'bee eater', 'red wolf, maned wolf, Canis rufus, Canis niger']
Monarch: ['coucal', 'agama', 'hornbill', 'bee eater', 'frilled lizard, Chlamydosaurus kingi']
True: coucal

Softmax: ['Mexican hairless', 'Italian greyhound', 'Doberman, Doberman pinscher', 'Great Dane', 'Weimaraner']
Monarch: ['Italian greyhound', 'Weimaraner', 'Mexican hairless', 'whippet', 'Great Dane']
True: Italian greyhound

Softmax: ['valley, vale', 'alp', 'lakeside, lakeshore', 'volcano', 'sandbar, sand bar']
Monarch: ['valley, vale', 'volcano', 'alp', 'lakeside, lakeshore', 'sandbar, sand bar']
True: volcano

Softmax: ['Welsh springer spaniel', 'Sussex spaniel', 'English springer, English springer spaniel', 'clumber, clumber spaniel', 'Brittany spaniel']
Monarch: ['Blenheim spaniel', 'Welsh springer spaniel', 'Brittany spaniel', 'cocker spaniel, English cocker spaniel, cocker', 'Sussex spaniel']
True: Welsh springer spaniel

Softmax: ['chickadee', 'junco, snowbird', 'jay', 'water ouzel, dipper', 'house finch, linnet, Carpodacus mexicanus']
Monarch: ['chickadee', 'cockroach, roach', 'cabbage butterfly', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'leafhopper']
True: chickadee

Monarch to Softmax FLOP ratio: 1.18e+08/2.98e+08 (39.48%)
```
