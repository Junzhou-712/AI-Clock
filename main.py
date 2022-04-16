from textRecognizer import TextRecognizer
from clockRecognizer import ClockRecognizer
import os

if __name__ == "__main__":
    cr = ClockRecognizer()
    img_name = "HD213.jpg"
    # 检查是否有rotate_
    #if not os.path.exists('./tmp' + os.sep + "rotate_" + img_name):
    cr.process(pathname="./cocome" + os.sep + img_name, save_result_img=True, show_img=True)

    img_name = "rotate_" + img_name

    tr = TextRecognizer()
    img_name = tr._modify('./tmp', img_name)
    tr_res = tr.detect('tmp', img_name, show_result=False)
    save_name = tr.save_path + os.sep + img_name.split('.')[0] + "_ocr_res.json"
    tr.save_res(tr_res, save_name)
    json_name = img_name.split(".")[0] + "_ocr_res.json"
    cr.read_degree("./ocr_json_file"+os.sep+json_name, "./tmp" + os.sep + img_name)
