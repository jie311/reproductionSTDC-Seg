#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random
import numpy as np


# 随机裁剪
class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size
    
    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size
        
        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
            im=im.crop(crop),
            lb=lb.crop(crop)
        )


# 随机水平翻转
class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p
    
    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im=im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb=lb.transpose(Image.FLIP_LEFT_RIGHT),
                        )


# 随机放缩
class RandomScale(object):
    def __init__(self, scales=(1,), *args, **kwargs):
        self.scales = scales
        # print('scales: ', scales)
    
    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        # scale = np.random.uniform(min(self.scales), max(self.scales))
        w, h = int(W * scale), int(H * scale)
        return dict(im=im.resize((w, h), Image.BILINEAR),
                    lb=lb.resize((w, h), Image.NEAREST),
                    )


# 色彩抖动??
class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        # 亮度
        if not brightness is None and brightness > 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        # 对比度
        if not contrast is None and contrast > 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        # 饱和度
        if not saturation is None and saturation > 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]
    
    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im=im,
                    lb=lb,
                    )


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales
    
    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W * ratio), int(H * ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class Compose(object):
    # 初始化时传入需要做的变换
    def __init__(self, do_list):
        self.do_list = do_list
    
    # 当再次调用时依次做函数变换
    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


if __name__ == '__main__':
    flip = HorizontalFlip(p=1)
    crop = RandomCrop((321, 321))
    rscales = RandomScale((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')
