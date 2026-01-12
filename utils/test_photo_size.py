from PIL import Image

path = '/root/autodl-tmp/mine-qr-code/mine_model_v2/dataset/train/lr/lr_08998.png'
with Image.open(path) as im:
    w, h = im.size          # (width, height)
    print(f'尺寸：{w} × {h} px')