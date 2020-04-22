import random

from PIL import Image, ImageDraw

for i in range(5000):
    img = Image.new('RGB', (100, 30), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10, 10), str(random.randint(0, 9999999)), fill=(255, 255, 255))
    img.save('../data/no_img/%s.png' % i)
