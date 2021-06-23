from csv import writer
import re
import pandas as pd
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import numpy as np
from stage2.segment import *
from PIL import Image
import pytesseract
import os
from flask import Flask, render_template, request, redirect
import sqlite3
import csv
app = Flask(__name__)

"""hyper parameters"""
if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False


def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    # m.print_network()
    m.load_weights(weightfile)

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
    project_path = os.getcwd()

    if os.path.exists(".\preds\predictions.jpg") and os.path.exists(".\preds\gray.png"):
        os.remove('.\preds\predictions.jpg')
        os.remove('.\preds\gray.png')
    result = plot_boxes_cv2(
        img, boxes[0], savename='preds\predictions.jpg', class_names=class_names)
    result = cv2.resize(result, (int(600), int(600)))
    # cv2.imshow("Bounding Box", result)
    # cv2.imwrite(".\preds\bbox.jpg", result)
    # cv2.waitKey(1)
    if os.path.exists('.\preds\predictions.jpg'):
        # segmentation('.\preds\predictions.jpg')image = cv2.imread(args["image"])
        img = cv2.imread('.\preds\predictions.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if args.p == "thresh":
            rect, gray = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif args.p == "blur":
            gray = cv2.medianBlur(gray, 3)

        cv2.imwrite('.\preds\gray.png', gray)
        text = pytesseract.image_to_string(Image.open('.\preds\gray.png'))
        if not os.path.exists(".\data.csv"):
            raw_data = {'date': [time.asctime(
                time.localtime(time.time()))], 'Number Plate': [text]}
            df = pd.DataFrame(raw_data)
            df.to_csv('data.csv', mode='a')
        else:
            df = pd.read_csv('.\data.csv')
            df.loc[len(df.index)] = [len(df.index), time.asctime(
                time.localtime(time.time())), text]
            df.to_csv('data.csv', mode='w', index=False)
        os.remove('.\preds\gray.png')
        # cv2.imshow("Image", img)
        # cv2.imshow("Output", gray)
        # cv2.waitKey(1)


def detect_cv2_camera(cfgfile, weightfile):
    args = get_args()
    import cv2
    m = Darknet(cfgfile)

    # m.print_network()
    m.load_weights(weightfile)

    if use_cuda:
        m.cuda()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./output.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)
    frame_count = 0
    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        # start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        # finish = time.time()

        result_img = plot_boxes_cv2(
            img, boxes[0], savename=None, class_names=class_names)

        # cv2.imshow('Yolo demo', result_img)
        if np.any(img != result_img):
            cv2.imwrite("./frames/frame{}.png".format(frame_count), result_img)
            frame_count += 1
        # cv2.waitKey(1)

    cap.release()


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    # m.print_network()
    m.load_weights(weightfile)

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(['Cars', 'Plate'])

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg',
                   class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser(
        'Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/darknet-yolov3.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./weight_folder/lapi.weights',
                        help='path of trained model.', dest='weightfile')
    # parser.add_argument('-imgfile', type=str,
    #                     default=None,
    #                     help='path of your image file.', dest='imgfile')
    parser.add_argument("-p", "--preprocess", dest='p', type=str, default="thresh",
                        help="type of preprocessing to be done")
    args = parser.parse_args()

    return args


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/accept/<number>')
def accept(number):
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM pass_application WHERE plate_no=?", (number,))
    rows = cur.fetchall()
    data = rows[0]
    conn.execute(
        "INSERT INTO pass_allowed VALUES(?,?,?,?,?,?,?,?,?)", data)
    conn.execute("DELETE FROM pass_application WHERE plate_no=?", (number,))
    conn.commit()
    conn.close()
    return redirect('/pass_application')


@ app.route('/reject/<number>')
def reject(number):
    conn = sqlite3.connect('database.db')
    conn.execute("DELETE FROM pass_application WHERE plate_no=?", (number,))
    conn.commit()
    conn.close()
    return redirect('/pass_application')


@ app.route('/pass_application')
def pass_application():
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM pass_application")
    rows = cur.fetchall()
    conn.close()
    if len(rows) == 0:
        flag = True
    else:
        flag = False
    return render_template('show-applications.html', rows=rows, flag=flag)


@app.route('/application_problem')
def application_problem():
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()

    cur.execute("SELECT * FROM plate_numbers")
    rows = cur.fetchall()
    problems = []
    for row in rows:
        cur.execute(
            "SELECT * FROM pass_allowed WHERE plate_no=?", (row[0],))
        rows1 = cur.fetchall()
        if len(rows1) == 0:
            problems.append(row[0])
    conn.commit()
    conn.close()
    if len(problems) == 0:
        flag = False
    else:
        flag = True
    return render_template('application_problem.html', problems=problems, flag=flag)


@ app.route('/store_info', methods=["POST"])
def store_info():
    req = request.form
    firstname = req.get("inputFirstName4")
    lastname = req.get("inputLastName4")
    email = req.get("inputEmail4")
    plate_no = req.get("inputPlateNo")
    address1 = req.get("address1")
    address2 = req.get("address2")
    inputCity = req.get("inputCity")
    inputState = req.get("inputState")
    inputZip = req.get("inputZip")
    values = (firstname, lastname, email, plate_no, address1,
              address2, inputCity, inputState, inputZip)
    conn = sqlite3.connect('database.db')
    conn.execute(
        "INSERT INTO pass_application VALUES(?,?,?,?,?,?,?,?,?)", values)
    conn.commit()
    conn.close()
    return redirect('/')


if __name__ == '__main__':
    if os.path.exists("data.csv"):
        os.remove("data.csv")
    args = get_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_list = os.listdir(dir_path+"\images")
    for img in file_list:
        detect_cv2(args.cfgfile, args.weightfile,
                   "C:/Users/Asus/Desktop/project_ma/Automatic-Number-Plate-Recognition/images/" + img)
    cv2.destroyAllWindows()
    conn = sqlite3.connect('database.db')
    print("Opened database successfully")
    with open('data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        line_count = 0
        for row in csv_reader:
            t = "".join(re.findall('[0-9A-Za-z]+', row[2]))
            conn.execute("INSERT INTO plate_numbers VALUES(?)", (t,))
        conn.commit()
        conn.close()
    app.run()
    # print(file_list)
    # if args.imgfile:
    #     detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
    # else:
    #     detect_cv2_camera(args.cfgfile, args.weightfile)
