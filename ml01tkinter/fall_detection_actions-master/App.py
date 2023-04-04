import os
import cv2
import time
import torch
import screeninfo
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import colorchooser
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def get_monitor_from_coord(x, y):  # multiple monitor dealing.
    monitors = screeninfo.get_monitors()
    for m in reversed(monitors):
        if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
            return m
    return monitors[0]


class Models:
    def __init__(self):
        self.inp_dets = 416
        self.inp_pose = (256, 192)
        self.pose_backbone = 'resnet50'
        self.show_detected = True
        self.show_skeleton = True
        self.device = 'cuda'

        self.load_models()

    def load_models(self):
        self.detect_model = TinyYOLOv3_onecls(self.inp_dets, device=self.device)
        self.pose_model = SPPE_FastPose(self.pose_backbone, self.inp_pose[0], self.inp_pose[1],
                                        device=self.device)
        self.tracker = Tracker(30, n_init=3)
        self.action_model = TSSTG(device=self.device)

    def kpt2bbox(self, kpt, ex=20):
        return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                         kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

    def process_frame(self, frame):
        detected = self.detect_model.detect(frame, need_resize=False, expand_bb=10)

        self.tracker.predict()
        for track in self.tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [1.0, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []
        if detected is not None:
            poses = self.pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
            detections = [Detection(self.kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]
            if self.show_detected:
                for bb in detected[:, 0:5]:
                    bb = list(map(int,bb.numpy()))
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        self.tracker.update(detections)
        for i, track in enumerate(self.tracker.tracks):
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = self.action_model.predict(pts, frame.shape[:2])
                action_name = self.action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)

                track.actions = out

            if track.time_since_update == 0:
                if self.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_DUPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        return frame


class main:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title('Human Falling Detection')
        self.master.protocol('WM_DELETE_WINDOW', self._on_closing)
        self.main_screen = get_monitor_from_coord(master.winfo_x(), master.winfo_y())

        self.recognitionFlg = True
        self.stop_video = False
        self.file_path = None
        self.width = int(self.main_screen.width * .85)
        self.height = int(self.main_screen.height * .85)
        self.master.geometry('{}x{}'.format(self.width, self.height + 15))

        toolbar = Frame(master)
        # 创建选择视频文件按钮
        self.choose_file = tk.Button(toolbar, text="选择视频文件", command=self.select_file)
        self.choose_file.pack(side="left")
        # 创建开始和停止识别按钮
        self.start_recognition = tk.Button(toolbar, text="开始识别", command=self.begin_recognition)
        self.start_recognition.pack(side="left")
        self.stop_recognition = tk.Button(toolbar, text="停止识别", command=self.end_recognition)
        self.stop_recognition.pack(side="left")
        # 创建打开和关闭摄像头按钮
        self.open_camera = tk.Button(toolbar, text="打开摄像头", command=self.activate_camera)
        self.open_camera.pack(side="left")
        self.close_camera = tk.Button(toolbar, text="关闭摄像头", command=self.deactivate_camera)
        self.close_camera.pack(side="left")
        toolbar.pack(side=TOP, fill=X)

        self.canvas_frame = Frame(self.master)
        self.cam = None
        self.canvas = tk.Canvas(self.canvas_frame, width=int(self.width * .65), height=self.height)
        self.canvas.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)

        fig = plt.Figure(figsize=(6, 8), dpi=100)
        fig.suptitle('Actions')
        self.ax = fig.add_subplot(111)
        self.fig_canvas = FigureCanvasTkAgg(fig, self.canvas_frame)
        self.fig_canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5, sticky=tk.NSEW)

        self.canvas_frame.pack(expand=True, fill="both")

        # Load Models
        self.resize_fn = ResizePadding(416, 416)
        self.models = Models()

        self.actions_graph()

        self.delay = 15
        #self.load_cam(r'D:\01/fall.mp4')
        self.update()

    def preproc(self, image):
        image = self.resize_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_cam(self, source):
        if self.cam:
            self.cam.__del__()

        if type(source) is str and os.path.isfile(source):
            self.cam = CamLoader_Q(source, queue_size=1000, preprocess=self.preproc).start()
        else:
            self.cam = CamLoader(source, preprocess=self.preproc).start()

    def actions_graph(self):
        if len(self.models.tracker.tracks) == 0:
            return
        track = self.models.tracker.tracks[0]
        if hasattr(track, 'actions'):
            y_labels = self.models.action_model.class_names
            self.ax.barh(np.arange(len(y_labels)), track.actions)
        self.fig_canvas.draw()

    def update(self):
        if self.cam is None:
            return
        if self.cam.grabbed():
            frame = self.cam.getitem()

            frame = self.models.process_frame(frame)

            frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()),
                               interpolation=cv2.INTER_CUBIC)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        else:
            self.cam.stop()

        self._cam = self.master.after(self.delay, self.update)

    def _on_closing(self):
        self.master.after_cancel(self._cam)
        if self.cam:
            self.cam.stop()
            self.cam.__del__()
        self.master.destroy()
    def select_file(self):
        self.recognitionFlg = False
        self.file_path = filedialog.askopenfilename(title="选择视频")
        #file_path = filedialog.askopenfilename(title="选择视频", filetypes=[("视频文件", "*.MP4;*.jpeg;*.png;*.bmp")])
        if self.file_path:
            print("选择视频：",self.file_path)
            self.video_capture = cv2.VideoCapture(self.file_path)
            ret, frame = self.video_capture.read()
            self.show_canvas(frame)
    def show_canvas(self, img_cv):
        frame= Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(frame)
        self.canvas.config(width=frame.size[0], height=frame.size[0])
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def begin_recognition(self):
        self.recognitionFlg = True
        i = 0
        while self.recognitionFlg:
            if self.video_capture.isOpened():
                i = i+1
                ret, frame = self.video_capture.read()
                if ret:
                    frame = recognition(frame)
                    self.show_canvas(frame)
                    self.update()
                    self.after(1)
                    cv2.waitKey(40)
                else:
                    break
        # 实现开始识别功能
        pass
    def end_recognition(self):
        self.recognitionFlg = False
        # 实现停止识别功能
        pass
    def activate_camera(self):
        # 实现打开摄像头功能
        self.video_capture = cv2.VideoCapture(0)
        # ret, frame = self.video_capture.read()
        # self.show_canvas(frame)
        self.recognitionFlg = False
        self.stop_video = False
        while not self.stop_video:
            if self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if ret:
                    if self.recognitionFlg:
                        frame = recognition(frame)
                    self.show_canvas(frame)
                    self.update()
                    self.after(1)
                    cv2.waitKey(40)
                else:
                    break
    def deactivate_camera(self):
        # 实现关闭摄像头功能
        self.stop_video = True
        self.image_show = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        self.photo = ImageTk.PhotoImage(self.image_show)
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.update()

root = tk.Tk()
app = main(root)
root.mainloop()
