#from textRecognizer import TextRecognizer
from clockRecognizer import ClockRecognizer
import os

if __name__ == "__main__":
    cr = ClockRecognizer()
    img_name = "HD213.jpg"
    # if not os.path.exists('./tmp/'+"rotate_"+img_name):
    cr.process(pathname="./cocome/"+img_name,save_result_img=True,show_img=True)
    # cr已经将图片保存到了./tmp
    
    json_name = "rotate_" + img_name.split(".")[0] + "_ocr_res.json"
    cr.read_degree(json_name, "./tmp"+os.sep+"rotate_"+img_name)
