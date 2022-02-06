import time

import cv2
import numpy as np

from time import strftime, perf_counter
from datetime import date, datetime

from tkinter import *
from tkinter.font import Font
from tkinter import messagebox

from PIL.Image import fromarray
from PIL.ImageTk import PhotoImage

from os import system
from threading import Thread
from imutils import resize
from pickle import loads

import serial


# 影像識別
class Face:
    # 初始化設定。
    def __init__(self):
        # 設置GUI介面權限
        self.window = Tk()
        # 設置GUI介面標題
        self.window.title('通訊與IOT專題實務')

        # 螢幕寬度
        self.screen_width = self.window.winfo_screenwidth()
        # 螢幕高度
        self.screen_height = self.window.winfo_screenheight()

        # 設置視窗置中
        screen_x = str(int((self.screen_width - 800) / 2))
        screen_y = str(int((self.screen_height - 600) / 2))

        # 設置GUI介面視窗大小為800x600
        self.window.geometry('800x600' + '+' + screen_x + '+' + screen_y)
        # 固定GUI介面視窗大小(不可調整)
        self.window.resizable(False, False)
        # 設置GUI介面視窗背景為白色
        self.window.configure(background='white')

        # ===== 以下這個區塊主要是設定臉部辨識的所需匯入的檔案 ===== #
        self.protoPath = 'face_detection_model/deploy.prototxt'
        self.modelPath = 'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel'
        self.detector = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)
        self.embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
        self.recognizer = loads(open("output/recognizer.pickle", "rb").read())
        self.le = loads(open("output/le.pickle", "rb").read())
        # ===== 以上這個區塊主要是設定臉部辨識的所需匯入的檔案 ===== #

        self.COM_PORT = 'COM3'  # 指定通訊埠名稱
        self.BAUD_RATES = 9600  # 設定傳輸速率
        self.ser = serial.Serial(self.COM_PORT, self.BAUD_RATES)  # 初始化序列通訊埠

        # 建立攝影機使用
        self.vid = MyVideoCapture(0)

        # 紀錄當前時間。
        self.now_time = datetime.now().strftime("%H:%M:%S")
        self.start_time = "08:30:00"
        self.end_time = "23:30:00"

        # 設定元件擺放。
        self.set_element()

        # 每1000毫秒(1秒)，更新臉部識別按鈕狀態。
        self.delay_state = 1000
        self.change_face_recognition_button_state()

        # 臉部識別按鈕事件(DISABLED：錯誤訊息欄、NORMAL：臉部辨識)。
        self.face_recognition_button.bind("<Button-1>", self.check_face_recognition_button)

        # 每15毫秒，更新相機畫面。
        self.delay_image = 15
        self.update_image()

        # 每1000毫秒(1秒)，更新時間設定。
        self.delay_clock = 1000
        self.update_clock()

        # 將影像辨識次數設置為0次。
        self.hint_flag = 0

        # 更新識別狀況列表。
        self.update_attend_status()

        self.RFID_ID = ''
        self.threading_MFRC522_read()
        self.add_RFID_ID_list()

        # 顯示GUI介面
        self.window.mainloop()

    # 設定元件樣式與位置。
    def set_element(self):
        # 設置標題為'影像識別：GUI介面設計'
        self.header_label = Label(self.window, text='影像識別：GUI介面設計', bg="white", fg="black", font=Font(size=20))
        self.header_label.place(x=200, y=10, width=400, height=40)

        # 建立日期標題
        self.date_label = Label(self.window, text='日期', bg="white", fg="black", font=Font(size=16))
        self.date_label.place(x=120, y=62.5, width=180, height=25)

        # 建立時間標題
        self.time_label = Label(self.window, text='時間', bg="white", fg="black", font=Font(size=16))
        self.time_label.place(x=500, y=62.5, width=180, height=25)

        # 設定臉部訓練按鈕的樣式與位置
        self.face_train_button = Button(self.window, text="臉部訓練", font=Font(size=16), bg='#F0F0F0',
                                        command=self.threading_face_train)
        self.face_train_button.place(x=80, y=470, width=120, height=40)

        # 設定臉部識別按鈕的樣式與位置
        self.face_recognition_button = Button(self.window, text="臉部識別", bg='#F0F0F0', font=Font(size=16))
        self.face_recognition_button.place(x=235, y=470, width=120, height=40)

        # 建立識別狀況列表
        self.attend_status_frame = Frame(self.window)
        self.attend_status_frame.place(x=430, y=130, width=310, height=310)

        self.delete_data_button = Button(self.window, text="刪除末筆資料", font=Font(size=14), bg='#F0F0F0',
                                         command=self.delete_data)
        self.delete_data_button.place(x=435, y=470, width=140, height=40)

        self.clear_select_data_button = Button(self.window, text="刪除選定資料", font=Font(size=14), bg='#F0F0F0',
                                               command=self.clear_select_data)
        self.clear_select_data_button.place(x=435, y=530, width=140, height=40)

        self.clear_all_data_button = Button(self.window, text="刪除所有資料", font=Font(size=14), bg='#F0F0F0',
                                            command=self.clear_all_data)
        self.clear_all_data_button.place(x=595, y=470, width=140, height=40)

        # 標註系級學號姓名
        self.name_ID = Label(self.window, text='電機系統三B0721251楊仁傑', bg="white", fg="black", font=Font(size=12))
        self.name_ID.place(x=595, y=565, width=200, height=40)

    # 顯示當前時間。
    def update_clock(self):
        week_day_dict = {0: '(一)', 1: '(二)', 2: '(三)', 3: '(四)',
                         4: '(五)', 5: '(六)', 6: '(日)', }

        # 讀取當前日期
        now_date, day = strftime("%Y-%m-%d"), date.today().weekday()

        # 設置當前日期
        now_date_info = Label(text=now_date + ' ' + week_day_dict[day], bg="white", fg="black", font=Font(size=16))
        now_date_info.place(x=120, y=90, width=180, height=25)

        # 讀取當前時間。
        now_time = strftime("%H:%M:%S")

        # 設置當前時間。
        now_time_info = Label(text=now_time, bg="white", fg="black", font=Font(size=16))
        now_time_info.place(x=500, y=90, width=180, height=25)

        self.window.after(self.delay_clock, self.update_clock)

    # 顯示相機畫面。
    def update_image(self):
        # 讀取相機畫面。
        ret, frame = self.vid.get_frame()

        # 如果有相機有成功運作，則執行以下區塊。
        if ret:
            # 將畫面尺寸調整為310*310。
            frame_resize = cv2.resize(frame, (310, 310))

            # 載入分類器。
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            # 將圖片轉為灰階圖片。
            gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)

            # 偵測臉部。
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # 繪製人臉部份的方框
            for (x, y, w, h) in faces:
                # (255, 255, 0)欄位可以變更方框顏色(紅，綠，藍)
                cv2.rectangle(frame_resize, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # 建立即時影像畫布。
            canvas_face = Canvas(self.window, width=310, height=310)
            canvas_face.place(x=60, y=130, width=310, height=310)

            # 顯示攝影機畫面。
            # 因canvas()函數設定問題，須將photo設置為global全域變數，否則無法正常顯示即時影像。
            global photo
            photo = PhotoImage(image=fromarray(frame_resize))
            canvas_face.create_image(1, 0, image=photo, anchor="nw")

        self.window.after(self.delay_image, self.update_image)

    # 臉部模型訓練。
    @staticmethod
    def face_train():
        system('python extract_embeddings.py\
                --dataset dataset\
                --embeddings output/embeddings.pickle\
                --detector face_detection_model\
                --embedding-model openface_nn4.small2.v1.t7')
        system('python train_model.py\
                --embeddings output/embeddings.pickle\
                --recognizer output/recognizer.pickle\
                --le output/le.pickle')

        # 顯示訓練成功訊息。
        messagebox.showinfo('狀態欄', '臉部模型訓練成功！')

    # 平行化運作臉部模型訓練。
    def threading_face_train(self):
        # 將face_train添加至子執行緒中
        process = Thread(target=self.face_train)
        # 子執行緒執行face_train
        process.start()

    # 更新識別狀況列表。
    def update_attend_status(self):
        # 建立出席狀況區框架裡的卷軸功能
        sb_status = Scrollbar(self.attend_status_frame)
        sb_status.pack(side=RIGHT, fill=Y)

        # 建立一個出席狀況清單
        self.attend_status_listbox = Listbox(self.attend_status_frame, height=8, width=32, yscrollcommand=sb_status.set,
                                             font=Font(size=14))
        self.attend_status_listbox.pack(side=LEFT, fill=BOTH)

    # 設置臉部識別按紐狀態。
    def change_face_recognition_button_state(self):
        # 可使用按鈕時間。
        # #000000：純黑色, #F0F0F0：淺灰色
        if self.start_time < self.now_time < self.end_time:
            self.face_recognition_button.config(fg='#000000', bg='#F0F0F0', state=NORMAL)
        # 不可使用按鈕時間。
        # #FFFFFF：純白色, #FF0000：純紅色
        else:
            self.face_recognition_button.config(fg='#FF0000', bg='#FF0000', state=DISABLED)

        self.window.after(self.delay_state, self.change_face_recognition_button_state)

    # 判斷臉部識別按鈕狀態(bind觸發事件)。
    def check_face_recognition_button(self, event):
        if self.face_recognition_button['state'] == DISABLED:
            messagebox.showerror("錯誤提醒", "現在並非臉部識別時間！\n辨識時間為：" + self.start_time + "~" + self.end_time)
        else:
            self.add_face_name_list()

    # 添加名字至識別狀況列表。
    def add_face_name_list(self):
        # 清空名字內容
        self.name = ''

        # 執行臉部識別
        self.face_recognition()

        # 中英名對照字典
        name_dict = {
            'Kuo Hsing-Chun': '郭婞淳',
            'Yang Yong-Wei': '楊勇緯',
            'Deng Yu-Cheng': '鄧宇成',
            'Tang Zhi-Jun': '湯智鈞',
            'Wei Jun-Heng': '魏均珩',
            'Luo Jia-Ling': '羅嘉翎',
            'Lin Yun-Ru': '林昀儒',
            'Zheng Yi-Jing': '鄭怡靜',
            'Chen Wen-hui': '陳玟卉',
            'Yang Ren-Chieh': '楊仁傑',
        }

        # 成功識別名字的話，執行以下區塊。
        if self.name != '':
            # 將英文名字對應至中文名字。
            try:
                name_value = name_dict[self.name]
            except KeyError:
                name_value = 'unknown'

            # 將辨識結果添加至識別狀況列表。
            self.attend_status_listbox.insert(END, '(FACE) ' + str(
                '%02d' % (self.attend_status_listbox.size() + 1)) + '. ' + name_value + ' ' +
                                              "{:.2f}%".format(self.proba) + ' ' + strftime("%H") + ':' + strftime(
                "%M") + ':' + strftime("%S"))

    # 刪除最後一筆資料。
    def delete_data(self):
        self.attend_status_listbox.delete(END)

    # 刪除選中資料。
    def clear_select_data(self):
        self.attend_status_listbox.delete(self.attend_status_listbox.curselection())

    # 刪除所有資料。
    def clear_all_data(self):
        self.attend_status_listbox.delete(0, END)

    # 人臉識別。
    def face_recognition(self):
        ret, frame = self.vid.get_frame()

        if ret:
            # 讀取照片並且對顏色進行調整
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 調整最大尺寸為600pixels
            frame = resize(frame, width=600)

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                              (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face self.detector to localize faces in the input image
            # 將照片放入模型中並且預測其結果。

            self.detector.setInput(imageBlob)
            detections = self.detector.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.75:
                    # compute the (x, y)-coordinates of the bounding box for the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI
                    face = frame[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue

                    # construct a blob for the face ROI,
                    # then pass the blob through our face embedding model to
                    # obtain the 128-d quantification of the face
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    self.embedder.setInput(faceBlob)
                    vec = self.embedder.forward()

                    # perform classification to recognize the face
                    preds = self.recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    self.proba = preds[j] * 100
                    self.name = self.le.classes_[j]

                    # draw the bounding box of the face along with the associated probability
                    face_text = "{}: {:.2f}%".format(self.name, self.proba)
                    y = startY - 10 if startY - 10 > 10 else startY + 10

                    face_name_label = Label(text=face_text, bg="white", fg='red', font=Font(size=12))
                    face_name_label.place(x=125, y=440, width=180, height=25)

    # RFID讀取
    def MFRC522_read(self):
        while True:
            while self.ser.in_waiting:
                self.RFID_ID = self.ser.readline().decode()
                print(self.RFID_ID)
                time.sleep(0.1)

    def threading_MFRC522_read(self):
        # 將face_train添加至子執行緒中
        process = Thread(target=self.MFRC522_read)
        # 子執行緒執行face_train
        process.start()

    # 將讀取到的ID加入至識別狀況列表中。
    def add_RFID_ID_list(self):
        if self.RFID_ID != '':
            RFID_ID_dict = {'15861943\r\n': '同學A',
                            'FCA7AD7F\r\n': '同學B',
                            '14C8ADE7\r\n': '同學C'}
            self.attend_status_listbox.insert(END, '(RFID) ' + str(
                '%02d' % (self.attend_status_listbox.size() + 1)) + '. ' + RFID_ID_dict[
                                                  self.RFID_ID] + ' ' + strftime(
                "%H") + ':' + strftime(
                "%M") + ':' + strftime("%S"))
            self.RFID_ID = ''

        self.window.after(2000, self.add_RFID_ID_list)


# 建立相機動作。
class MyVideoCapture:
    def __init__(self, video_source):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.hint_teacher_check = 0

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return None, None

    # Release the video source when the object is destroyed
    def __del__(self):
        self.vid.release()


if __name__ == '__main__':
    Face()