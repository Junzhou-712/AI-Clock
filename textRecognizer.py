from paddleocr import PaddleOCR, draw_ocr
import json
import os
import cv2
import numpy as np

class TextRecognizer:
    def __init__(self):
        # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
        # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
        self.save_path = "./ocr_json_file"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _modify(self, imgpath, img_name):
        img = cv2.imread(imgpath + os.sep + img_name)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        img = cv2.filter2D(img, -1, kernel=kernel)
        cv2.imwrite(imgpath + os.sep + "modified_" + img_name, img)
        return "modified_"+img_name

    def detect(self, img_path, img_name, show_result):
        result = self.ocr.ocr(img_path+os.sep+img_name, cls=True)
        if show_result:
            for line in result:
                print(line)
        return result

    def save_res(self, tr_res, save_name):
        tr_res_dict = {}
        for res in tr_res:
            res_str = res[-1][0]
            try:
                res_num = float(res_str)
            except ValueError:
                continue
            res_dic = {}
            res_dic["p1"] = res[0][0]
            res_dic["p2"] = res[0][1]
            res_dic["p3"] = res[0][2]
            res_dic["p4"] = res[0][3]
            res_dic["acc"] = float(res[-1][-1])
            tr_res_dict[res_num] = res_dic
            
        with open(save_name, 'w', encoding='utf-8') as f:
            json.dump(tr_res_dict, f, ensure_ascii=False)


if __name__ == "__main__":
    img_name = "rotate_" + "HD517.jpg"

    tr = TextRecognizer()
    img_name = tr._modify('./tmp', img_name)
    tr_res = tr.detect('tmp', img_name, show_result=False)
    save_name = tr.save_path + os.sep + img_name.split('.')[0] + "_ocr_res.json"
    tr.save_res(tr_res, save_name)

'''
[2022/04/09 19:36:24] root INFO: [[[386.0, 311.0], [428.0, 311.0], [428.0, 350.0], [386.0, 350.0]], ('40', 0.99508715)]
[2022/04/09 19:36:24] root INFO: [[[584.0, 314.0], [625.0, 314.0], [625.0, 348.0], [584.0, 348.0]], ('60', 0.98909414)]
[2022/04/09 19:36:24] root INFO: [[[491.0, 408.0], [517.0, 408.0], [517.0, 428.0], [491.0, 428.0]], ('0', 0.67648894)]
[2022/04/09 19:36:24] root INFO: [[[677.0, 464.0], [714.0, 464.0], [714.0, 497.0], [677.0, 497.0]], ('30', 0.8265178)]
[2022/04/09 19:36:24] root INFO: [[[458.0, 628.0], [541.0, 628.0], [541.0, 669.0], [458.0, 669.0]], ('SSM', 0.99830437)]
[2022/04/09 19:36:24] root INFO: [[[592.0, 642.0], [662.0, 642.0], [662.0, 691.0], [592.0, 691.0]], ('100', 0.9994869)]
[2022/04/09 19:36:24] root INFO: [[[454.0, 674.0], [537.0, 668.0], [540.0, 713.0], [457.0, 720.0]], ('MB', 0.78069204)]
[2022/04/09 19:36:24] root INFO: [[[421.0, 786.0], [564.0, 791.0], [563.0, 822.0], [420.0, 817.0]], ('N0:302246', 0.8987596)]
'''