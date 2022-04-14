# 环境配置

+ conda环境
    ```sh
    conda create -n open-mmlab python=3.8 -y && conda activate open-mmlab
    ```

+ 先装paddlepaddle和paddleocr
    ```sh
    python3 -m pip install paddlepaddle-gpu==2.0.0 paddleocr==2.2.0
    ```

+ 再装pytorch， mmdetection
    ```
    conda install pytorch torchvision -c pytorch
    ```

    ```
    python3 -m pip install openmim==0.1.5 && mim install mmdet
    ```

<!-- 
+ 如果matplotlib冲突，降级labelme `python3 -m pip install labelme==4.2.0`

+ 如果缺wrapt `python3 -m pip install wrapt`

+ `Click`包可能有冲突但不影响 -->

## TODO
1. 简化代码整体逻辑、复杂度
2. 最小二乘法拟合圆的取样方法(轮廓列表好像不一定连续)
3. ocr度数后处理
    + 遮挡 0.16 -> 16
    + 其他数字标识，比如ID


## 备忘
+ 图片坐标系是width 为x, height为y; cv2.circle的参数是(x,y)也就是(w, h); bbox 的[x, y, x, y]也就是[w, h, w, h]