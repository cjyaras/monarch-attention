Applying Monarch-structured attention to base ViT on ImageNet, we can reduce FLOPs in the **attention operation** by roughly 60% with no loss in accuracy and without any additional training. 

To test on a few images, run
```
python examples/vit/test_monarch.py
```
from the top directory. Expected output:
```
URL: http://images.cocodataset.org/val2017/000000039769.jpg
Softmax: Egyptian cat | Monarch: Egyptian cat

URL: https://farm7.staticflickr.com/6139/6023621033_e4534f0655_z.jpg
Softmax: airliner | Monarch: airliner

URL: https://farm4.staticflickr.com/3699/8943306698_ca46820139_z.jpg
Softmax: tennis ball | Monarch: tennis ball

URL: https://farm3.staticflickr.com/2826/9688908056_22512acdaf_z.jpg
Softmax: plate | Monarch: plate

URL: https://farm2.staticflickr.com/1349/4610248959_30c464a5b6_z.jpg
Softmax: bagel, beigel | Monarch: bagel, beigel

URL: https://farm7.staticflickr.com/6140/5926597200_ae3122bcaa_z.jpg
Softmax: lakeside, lakeshore | Monarch: lakeside, lakeshore

Monarch to Softmax FLOP ratio: 1.41e+08/3.58e+08 (39.48%)
```