

# 环境安装

```
conda create -n open-mmlab python=3.8 -y && conda activate open-mmlab
```

```
conda install nomkl
```

安装torch
```
conda install pytorch torchvision -c pytorch
```

安装mmdetection
```
python3 -m pip install openmim
mim install mmdet
```

安装paddlepaddle

```sh
python3 -m pip install paddlepaddle-gpu==2.0.0 -i https://mirror.baidu.com/pypi/simple
```
安装padlleocr

```sh
python3 -m pip install "paddleocr==2.2.0" # 此版本不冲突
```

```
Click==7.1.2
flask==2.0

```

## todo

1.简化代码整体逻辑、复杂度
2.最小二乘法拟合圆的取样方法(轮廓列表好像不一定连续)
3.ocr度数后处理


# other
坐标系是width 为x, height为y; cv2.circle的参数是(x,y)也就是(w, h)
bbox 的[x, y, x, y]也就是[w, h, w, h]