# ViTMem

This package uses vision transformers in pytorch to estimate image memorability.

## How to use

Install from the python package index

```shell
pip install vitmem
```

Estimate image memorability by providing a filename

```python
from vitmem import ViTMem
model = ViTMem()
memorability = model("image.jpg")
print(f"Estimated memorability: {memorability}")
```

Estimate image memorability by providing a PIL Image object.

```python
from PIL import Image
from vitmem import ViTMem
model = ViTMem()
image = Image.open("image.jpg")
memorability = model(image)
print(f"Estimated memorability: {memorability}")
```

Estimate image memorability by providing a transformed image tensor:

```python
from PIL import Image
from vitmem import transform
from vitmem import ViTMem
model = ViTMem()
image = Image.open("image.jpg")
tensor = transform(image)
memorability = model(tensor)
print(f"Estimated memorability: {memorability}")
```

The model will attempt to use GPU/CUDA if available. 