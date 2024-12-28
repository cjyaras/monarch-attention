Applying Monarch-structured attention to base ViT on ImageNet, we can reduce FLOPs in the attention operation by roughly 60% with no loss in accuracy and without any additional training. 

To test on a few images, run
```
python examples/vit/test_monarch.py
```
from the top directory. 