from textRecognizer import TextRecognizer
from clockRecognizer import ClockRecognizer
import os
import sqlite3
import shutil
import time
import base64
if __name__ == "__main__":
    # 服务器
    img_dir = '/pidev/test'
    img_done_dir = '/pidev/done'
    db_path = '/home/joesu/test.db'

    if not os.path.exists(img_done_dir):
        os.makedirs(img_done_dir)
    if not os.path.exists("cocome"):
        os.makedirs("cocome")
    cr = ClockRecognizer()
    tr = TextRecognizer()

    id = 0
    while True:
        conn = sqlite3.connect(db_path)
        curs = conn.cursor()

        curs.execute('''
        CREATE TABLE IF NOT EXISTS CLOCK(
        id INT PRIMARY KEY, result FLOAT, raw_img LONGBOLB, rotate_img LONGBOLB, marked_img LONGBOLB, result_img LONGBOLB
        )
        ''')

        img_list = os.listdir()
        for img_name in img_list:
            shutil.move(img_dir+os.sep+img_name, "./cocome")
            origin_img_name = img_name
            cr.process(pathname="./cocome" + os.sep + img_name, save_result_img=True, show_img=True)
            img_name = "rotate_" + img_name
            img_name = tr._modify('./tmp', img_name)
            tr_res = tr.detect('tmp', img_name, show_result=False)
            save_name = tr.save_path + os.sep + img_name.split('.')[0] + "_ocr_res.json"
            tr.save_res(tr_res, save_name)
            json_name = img_name.split(".")[0] + "_ocr_res.json"
            result_degree = cr.read_degree("./ocr_json_file"+os.sep+json_name, "./tmp" + os.sep + img_name) 
            with open("./cocome" + os.sep + origin_img_name, "rb") as f:
                raw_bt = base64.b64encode(f.read())
            with open("./tmp" + os.sep + img_name, "rb") as f:
                rotate_bt = base64.b64encode(f.read())
            with open("./Inference_result" + os.sep + "img" + os.sep + "marked_"+origin_img_name, "rb") as f:
                marked_bt = base64.b64encode(f.read())
            with open("./Inference_result" + os.sep + "img" + os.sep + "result_"+origin_img_name, "rb") as f:
                result_bt = base64.b64encode(f.read())
            curs.execute("INSERT INTO CLOCK VALUES(?)",(id, result_degree, raw_bt, rotate_bt, marked_bt, result_bt))
            conn.commit()
            
            print("-------------------------------------------------------")
            print(img_dir+os.sep+img_name, "done. ")
            print(">> move ", img_dir+os.sep+img_name, " to ", img_done_dir)
            print("-------------------------------------------------------")
        curs.close()
        conn.close()
        time.sleep(3)
    
