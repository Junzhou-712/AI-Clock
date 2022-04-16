# from ctypes import pointer
# from curses import echo
# from importlib.resources import path
# from logging.handlers import RotatingFileHandler
from cv2 import imread
from mmdet.apis import init_detector, inference_detector
from mmdet.core.mask.structures import bitmap_to_polygon
import mmcv
import os
import numpy as np
import cv2
from math import atan, sqrt, pi, sin, cos, degrees, radians
import json

class ClockRecognizer:
    def __init__(self):
        self.config_file = './configs/myConfigs/pointer_config.py'
        self.checkpoint_file = './checkpoints/new.pth'
        self.model = init_detector(self.config_file, self.checkpoint_file, device='cuda:0')
        self.save_path = './Inference_result'
        self.min_area = 100
        self.min_point_distance = 100
        self.tmp_dir = './tmp'
        self.corner_config = {
            'block_size': [15, 80],
            'max_corners': [3, 2],
            'min_distance': [100, 300]
        }
        # 指针直线
        self.line_para = None
        # 指针直线方向
        self.pointer_direc_k = None

        # 刻度的端点坐标[[x0,y0],[x1,y1],[中点坐标]]
        self.scale_pointer = []
        # 表盘圆[圆心x，圆心y，半径]
        self.clock_circle = []
        #表盘度数的角度
        self.degree_dict = {}
        self._init_dir()

    def _init_dir(self):
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.save_path + os.sep + "img"):
            os.makedirs(self.save_path + os.sep + "img")

    def _line(self, para, x):
        k, b = para[0], para[1]
        return k * x + b

    def _distance_to_line(self, para, x, y):
        k, b = para
        return abs(k*x+b-y)/sqrt(k*k+1)

    def _distance_to_point(self, x1, y1, x2, y2):
        return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

    def _modify(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _rotate(self, image, mask, p0, p1, px):
        k = (p0[1] - p1[1])*1.0 / (p0[0] - p1[0])
        b = p0[1] - k*p0[0]
        p0 = np.array(p0+[1]).reshape(-1, 1)
        p1 = np.array(p1+[1]).reshape(-1, 1)
        angle = atan(k)*180/3.14
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        if px[0]*k - px[1] + b <= 0:
            angle += 180

        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        p0_ = np.dot(M, p0)
        p1_ = np.dot(M, p1)

        self.scale_pointer = [[p0_[0], p0_[1]], [p1_[0], p1_[1]], [(p0_[0]+p1_[0])/2, (p0_[1]+p1[1])/2]]
        return cv2.warpAffine(image, M, (nW, nH)), cv2.warpAffine(mask, M, (nW, nH))
    
    def _perspective_trans(self, rotate_img, mask):
        east = [0, 0]
        west = [0x3fffffff, 0x3fffffff]
        north = [0x3fffffff, 0x3fffffff]

        contours, _ = bitmap_to_polygon(mask)
        contour = contours[0]
        north = min(contour, key=lambda item: item[1])
        east = max(contour, key=lambda item: item[0])
        west = min(contour, key=lambda item: item[0])

        mid_height = (east[1] + west[1]) / 2
        east[1] = west[1] = mid_height
        mid_width = (self.scale_pointer[2][0][0] + north[0] + east[0] + west[0])/4
        north[0] = mid_width
        south = [mid_width, east[1]+west[1]-north[1]]

        points1 = np.float32([east, south, west, north])
        points2 = np.float32([[800, 500], [500, 800], [200, 500], [500, 200]])
        M = cv2.getPerspectiveTransform(points1, points2)
        # # 实现透视变换转换
        rotate_img = cv2.warpPerspective(rotate_img, M, (1500, 1500))
        # mask = cv2.warpPerspective(mask, M, (1500, 1500))
        return rotate_img, mask

    def _cornerDetect(self, image, origin_img, max_corners, min_distance, blockSize):
        quality_level = 0.01
        corners = cv2.goodFeaturesToTrack(image, max_corners, quality_level, min_distance, blockSize = blockSize)
        for pt in corners:
            origin_img = cv2.circle(origin_img, (np.int32(pt[0][0]), np.int32(pt[0][1])), 10, (255, 255, 0), 2)
        # output
        return corners, image, origin_img

    def calCircleCenter(self, pointsList):  # 最小二乘法计算拟合圆
        X1 = Y1 = X2 = Y2 = X3 = Y3 = X1Y1 = X1Y2 = X2Y1 = 0
        for point in pointsList:
            X1 += point[0]
            Y1 += point[1]
            X2 += point[0] * point[0]
            Y2 += point[1] * point[1]
            X3 += point[0] * point[0] * point[0]
            Y3 += point[1] * point[1] * point[1]
            X1Y1 += point[0] * point[1]
            X1Y2 += point[0] * point[1] * point[1]
            X2Y1 += point[0] * point[0] * point[1]

        N = len(pointsList)
        C = N * X2 - X1 * X1
        D = N * X1Y1 - X1 * Y1
        E = N * X3 + N * X1Y2 - (X2 + Y2) * X1
        G = N * Y2 - Y1 * Y1
        H = N * X2Y1 + N * Y3 - (X2 + Y2) * Y1
        a = (H * D - E * G) / (C * G - D * D)
        b = (H * C - E * D) / (D * D - G * C)
        c = -(a * X1 + b * Y1 + X2 + Y2) / N

        A = a / (-2)
        B = b / (-2)
        R = sqrt(a * a + b * b - 4 * c) / 2

        return A, B, R

    def _get_rotate_img(self, img, pathname, save_result_img, show_img):
        print('=====================================================')
        print('>> Start getting rotated img')
        print('>> Loading ', pathname, img)

        raw_img = cv2.imread(pathname + os.sep + img)
        gray_img = self._modify(raw_img)
        cv2.imwrite(self.tmp_dir + os.sep + img, gray_img)

        result = inference_detector(self.model, self.tmp_dir + os.sep + img)
        if save_result_img:  # 不加out_file参数可以显示图片
            self.model.show_result(
                pathname + os.sep + img,
                result,
                out_file=self.save_path + os.sep + 'img' + os.sep + 'result_0' + img
            )
        # result = [bbox_result, segm_result]; result[0]是目标box的二点坐标和置信度, 有两个，指针和刻度
        bbox_result, segm_result = result
        segms = mmcv.concat_list(segm_result)  # segms转化成list
        segms = np.stack(segms, axis=0)  # 把list segms => (2, height, width)

        # ----------------------------------------------------------------------
        pointer_mask_p = None
        mask = segms[0]
        mask = mask.copy().astype(np.uint8)

        # 取指针上的一点作为旋转方向参考
        bbox = bbox_result[0][0]
        pointer_mask_p = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]

        # ----------------------------------------------------------------------
        # 求刻度轮廓寻找角点
        mask = segms[1]  # segms[0]是指针
        mask = mask.copy().astype(np.uint8)

        # 得到角点坐标
        raw_img = imread(self.tmp_dir + os.sep + img)
        corner_lis, mask, raw_img = self._cornerDetect(
            mask,
            raw_img,
            max_corners=self.corner_config['max_corners'][1],
            min_distance=self.corner_config['min_distance'][1],
            blockSize=self.corner_config['block_size'][1]
        )
        print("corners: ", corner_lis)
        x0, y0, x1, y1 = corner_lis[0][0][0], corner_lis[0][0][1], corner_lis[1][0][0], corner_lis[1][0][1]
        rotate_img, mask = self._rotate(raw_img, mask, [x0, y0], [x1, y1], pointer_mask_p)

        bbox = bbox_result[1][0]
        rotate_img, mask = self._perspective_trans(rotate_img, mask)
        # 保存已经旋转好，透视变换好的图片
        cv2.imwrite(self.tmp_dir + os.sep + "rotate_" + img, rotate_img)
        print('=====================================================')

    # 得到指针所在直线和表盘拟合的圆
    def _get_circle(self, img, save_result_img, show_img):
        print('=====================================================')
        print('>> Start finding pointer line')
        pathname = self.tmp_dir + os.sep + "rotate_" + img
        print('>> Loading ', pathname)

        result = inference_detector(self.model, pathname)
        if save_result_img:  # 不加out_file参数可以显示图片
            self.model.show_result(
                pathname,
                result,
                out_file=self.save_path + os.sep + 'img' + os.sep + 'result_' + img
            )
        rotate_img = cv2.imread(pathname)

        bbox_result, segm_result = result
        segms = mmcv.concat_list(segm_result)  # segms转化成list
        segms = np.stack(segms, axis=0)  # 把list segms => (2, height, width)
        
        # 先得到圆心
        mask = segms[1]
        mask = mask.copy().astype(np.uint8)

        #  得到轮廓坐标
        contours, _ = bitmap_to_polygon(mask)
        corner_lis, mask, raw_img = self._cornerDetect(
            mask,
            rotate_img,
            max_corners=self.corner_config['max_corners'][1],
            min_distance=self.corner_config['min_distance'][1],
            blockSize=self.corner_config['block_size'][1]
        )
        x0, y0, x1, y1 = corner_lis[0][0][0], corner_lis[0][0][1], corner_lis[1][0][0], corner_lis[1][0][1]
        self.scale_pointer = [[x0, y0], [x1, y1], [(x0+y0)/2, (x1+y1)/2]]

        contour = contours[0]
        c1_lis, c2_lis = [], []
        index = 0
        cnt = 0
        cnt_color = [(255, 0, 255), (0, 255, 255), (255, 0, 255)]
        added = False
        # vis = [False] * len(contour)

        while True:
            vec = contour[index]
            x, y = vec[0], vec[1]
            index = (index + 20) % len(contour)
            d1 = self._distance_to_point(x, y, self.scale_pointer[0][0], self.scale_pointer[0][1])
            d2 = self._distance_to_point(x, y, self.scale_pointer[1][0], self.scale_pointer[1][1])
            if d1 <= self.min_point_distance or d2 <= self.min_point_distance:
                if added is False:
                    added = True
                    cnt += 1
                continue
            added = False
            if cnt == 0 or cnt == 2:
                if cnt == 2 and index > 0:
                    break
                c1_lis.append([x, y])
            elif cnt == 1:
                c2_lis.append([x, y])
            else:
                break
            if save_result_img:
                rotate_img = cv2.circle(rotate_img, (np.int32(x), np.int32(y)), 10, cnt_color[cnt], 5)

        print("c1_lis: ", len(c1_lis), "c2_lis: ", len(c2_lis))
        A1, B1, R1 = self.calCircleCenter(c1_lis)
        A2, B2, R2 = self.calCircleCenter(c2_lis)
        self.clock_circle = [(A1+A2)/2, (B1+B2)/2, (0.8*R1+0.2*R2) if (R1 > R2) else (0.8*R2+0.2*R1)]
        print("clock_circle: ", self.clock_circle)

        mask = segms[0]
        mask = mask.copy().astype(np.uint8)
        # 寻找指针直线--最小二乘法
        bbox = bbox_result[0][0]
        x_coord, y_coord = [], []
        for row in range(np.int32(bbox[0]), np.int32(bbox[2])):
            for col in range(np.int32(bbox[1]), np.int32(bbox[3])):
                if mask[row][col] > 0:
                    mask[row][col] = 255
                    x_coord.append(col)
                    y_coord.append(row)

        ones = np.ones(len(x_coord)).reshape(-1, 1)
        x_coord = np.array(x_coord).reshape(-1, 1)
        x_coord = np.concatenate((x_coord, ones), axis=1)
        y_coord = np.array(y_coord)
        para1 = np.dot(np.dot(np.linalg.inv(np.dot(x_coord.T, x_coord)), x_coord.T), y_coord)

        same_cnt = 0
        for x in range(np.int32(bbox[0]), np.int32(bbox[2])):
            for y in range(np.int32(bbox[1]), np.int32(bbox[3])):
                if mask[x][y] > 0:
                    vx = x - self.clock_circle[0]
                    vy = y - self.clock_circle[1]
                    if para1[0] >= 0:
                        if vx >= 0 and vy >= 0:
                            same_cnt += 1
                        elif vx <= 0 and vy <= 0:
                            same_cnt -= 1
                    elif para1[0] < 0:
                        if vx >= 0 and vy <= 0:
                            same_cnt += 1
                        elif vx <= 0 and vy >= 0:
                            same_cnt -= 1

        self.line_para = [para1[0], para1[1], same_cnt]

        if save_result_img:
            rotate_img = cv2.circle(rotate_img, (np.int32(self.clock_circle[0]), np.int32(self.clock_circle[1])), np.int32(self.clock_circle[2]), (255, 0, 0), 5)
            rotate_img = cv2.line(rotate_img, (0, np.int32(self._line(self.line_para, 0))), (rotate_img.shape[0], np.int32(self._line(self.line_para, rotate_img.shape[0]))), (255, 0, 0), 10)
            rotate_img = cv2.circle(rotate_img, (np.int32(bbox[0]), np.int32(bbox[1])), 10, (255, 0, 0), 2)
            rotate_img = cv2.circle(rotate_img, (np.int32(bbox[2]), np.int32(bbox[3])), 10, (255, 0, 0), 2)
            cv2.imwrite(self.save_path + os.sep + "img" + os.sep + "marked_" + img, rotate_img)
        print('=====================================================')

    def post_process(self, sort_lis):
        """
        1. 筛掉角度过小或过大
        2. 圆心距离过小
        3.
        """
        temp_lis1 = sort_lis
        temp_lis2 = []
        min_scale = 0x3fffffff
        min_scale_degree = 0

        for item in temp_lis1:
            scale, (degree, center_dis) = item
            if degree < 45 or degree > 315:
                continue
            if center_dis < self.clock_circle[2]*0.3:
                continue
            if scale < min_scale:
                min_scale = scale
                min_scale_degree = degree
            temp_lis2.append((scale, (degree, center_dis)))
        temp_lis1 = []
        min_scale_divide_degree = 0x3fffffff
        for item in temp_lis2:
            scale, (degree, center_dis) = item
            if degree < min_scale_degree:
                while scale > min_scale:
                    scale /= 10
                temp_lis1.append((scale, (degree, center_dis)))
                continue

            min_scale_divide_degree = min(min_scale_divide_degree, scale*1.0/(degree-35))
            if degree > min_scale_degree:
                while scale*1.0/(degree-35) > 2*min_scale_divide_degree:
                    print(scale, min_scale_divide_degree)
                    scale /= 10
                    temp_lis1.append((scale, (degree, center_dis)))
                    continue
            
            temp_lis1.append((scale, (degree, center_dis)))

        res_lis = temp_lis1
        # 补全刻度0--最小二乘法
        # deg = zero + num * k
        num_coord = []
        deg_coord = []
        if res_lis[0][0] != 0:
            for item in res_lis:
                num_coord.append(item[0])
                deg_coord.append(item[1])
        ones = np.ones(len(num_coord)).reshape(-1, 1)
        num_coord = np.array(num_coord).reshape(-1, 1)
        num_coord = np.concatenate((num_coord, ones), axis=1)
        deg_coord = np.array(deg_coord)
        zero = np.dot(np.dot(np.linalg.inv(np.dot(num_coord.T, num_coord)), num_coord.T), deg_coord)[1][0]
        res_lis.insert(0, (0, (zero, 0)))
        print(res_lis, type(res_lis))
        return res_lis




    def read_degree(self, json_name, pathname, show_img):
        print('=====================================================')
        print('>> Start Reading Degree.')
        # 以圆心中垂线为0度
        with open(json_name, "r", encoding='utf-8') as f:
            ocr_res = json.load(f)

        rotate_img = cv2.imread(pathname)
        sort = sorted(ocr_res)
        for key in sort:
            try:
                res_num = float(key)
            except ValueError:
                continue
            cx = (ocr_res[key]["p1"][0]+ocr_res[key]["p2"][0]+ocr_res[key]["p3"][0]+ocr_res[key]["p4"][0])/4
            cy = (ocr_res[key]["p1"][1]+ocr_res[key]["p2"][1]+ocr_res[key]["p3"][1]+ocr_res[key]["p4"][1])/4
            k = (cy-self.clock_circle[1])/(cx-self.clock_circle[0])
            degree = degrees(atan(k))
            vx = cx - self.clock_circle[0]
            vy = cy - self.clock_circle[1]
            ex = cos(radians(degree))
            ey = sin(radians(degree))
            # 判断需不需要加180度
            if vx*ex + vy*ey < 0:
                degree += 180
            degree = (degree + 270) % 360
            rotate_img = cv2.circle(rotate_img, (np.int32(cx), np.int32(cy)), 10, (255, 0, 0), 5)
            d_center = sqrt(vx*vx+vy*vy)
            self.degree_dict[res_num] = (degree, d_center)

        lk, lcnt = self.line_para[0], self.line_para[2]

        degree = atan(lk)*180/pi
        # 投票输了
        if lcnt < 0:
            degree += 180
        degree = (degree + 270) % 360
        print(">> THE POINTER DEGREE: ", degree)
        print(self.degree_dict)
        # 得到一个list[(degree:angle)()]
        sort = sorted(self.degree_dict.items(), key=lambda x: x[1])
        sort = self.post_process(sort)
        degree_result = 0

        for i in range(len(sort)-1):
            if degree < sort[i+1][1][0]:
                degree_result = (sort[i+1][0]-sort[i][0])*(degree-sort[i][1][0])*1.0/(sort[i+1][1][0]-sort[i][1][0])+sort[i][0]
                break
            else:
                print(">>[ERROR!] The Degree errors!")
        print(">>> THE FINAL DEGREE RESULT: ", degree_result if degree_result > 0 else 0)

        if show_img:
            rotate_img = cv2.resize(rotate_img, (900, 900))
            cv2.imshow("final", rotate_img)
            cv2.waitKey(0)

    def process(self, pathname="", process_num=1, save_result_img=False, show_img=False):
        if os.path.isdir(pathname):
            filelist = os.listdir(pathname)
            cnt = 0
            for img in filelist:
                cnt += 1
                if cnt > process_num:
                    break
                self._get_rotate_img(img, pathname, save_result_img, show_img)
                self._get_circle(img, save_result_img, show_img)
        elif os.path.isfile(pathname):
            img = pathname.split('/')[-1]
            _pathname = ""
            path_lis = pathname.split('/')[:-1]
            cnt = 0
            for p in path_lis:
                if cnt == 0 and pathname[0] != '/':
                    pass
                else:
                    _pathname += '/'
                _pathname += p
                cnt+=1
            self._get_rotate_img(img, _pathname, save_result_img, show_img)
            self._get_circle(img, save_result_img, show_img)
        else:
            print("wtf?")


if __name__ == "__main__":
    c = ClockRecognizer()
    c.process(
        pathname="./cocome/HD213.jpg",
        save_result_img=True,
        show_img=True
    )


# # 测试视频并展示结果
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)