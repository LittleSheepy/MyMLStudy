from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QStackedWidget, QTabWidget
from PyQt5.QtWidgets import QPushButton, QLineEdit, QComboBox, QProgressBar, QAction, QRadioButton
from PyQt5.QtCore import Qt
from classify.train import run
import multiprocessing
from PyQt5.QtWidgets import QTextEdit, QLabel, QFileDialog
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import QObject
import sys
import os
from tqdm import tqdm
import time
import re
import subprocess
import webbrowser
from tensorboard import program
from utils.create_dataset import create_dataset
class Worker(QThread):
    def run(self):
        print("创建临时数据集")
        dir_src = window.centralWidget.lineEdit_data.text()
        class_nums = int(window.centralWidget.lineEdit_data_cnt.text())
        # class_nums = int(window.centralWidget.lineEdit_data_train.text())
        val_per = int(window.centralWidget.lineEdit_data_val.text())
        test_per = int(window.centralWidget.lineEdit_data_test.text())
        create_dataset(dir_src, "./tmp_datasets/", class_nums, val_per, test_per)
        data_name = os.path.basename(os.path.normpath(dir_src))
        data_path = os.path.join("./tmp_datasets/", data_name)
        img_size = int(window.centralWidget.comboBox.currentText())

        print("开始训练")
        epochs = int(window.centralWidget.lineEdit.text())
        # lineEdit_data = window.lineEdit_data.text()
        run(pyqt=window.centralWidget, epochs=epochs, data=data_path, imgsz=img_size) # , data=lineEdit_data

class TensorboardRun(QThread):
    def run(self):
        print("开始Tensorboard")
        # 指定exe文件的路径和参数
        current_path = os.getcwd()
        print(current_path)
        exe_path = './runTensorBoard.exe'
        if not os.path.exists(exe_path):
            print("没有文件 ", exe_path)
        exe_path = current_path + './runTensorBoard.exe'
        if not os.path.exists(exe_path):
            print("没有文件 ", exe_path)
            return
        args = []
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        # 打开exe文件
        subprocess.Popen([exe_path] + args)
def parse_progress(text):
    match = re.search(r'(\d+)%', text)
    if match:
        return int(match.group(1))
    else:
        return 0
class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))
    def flush(self):
        pass
class MyWindow(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        # self.setWindowTitle('KADO AI训练 v1.0.23.0626')
        # self.resize(800, 600)
        self.tabWidget = QTabWidget(self)

        # self.init_menuBar()
        self.init_tab_train()
        self.init_tab_creat_data()
        self.init_tab_infrence()
        self.init_tab_test()

        self.lineEdit.setText('100')
        self.lineEdit_data.setText(r'./datasetstest/')

        # Create a Stream object and set it as the new output

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.tabWidget)

        self.stream = Stream()
        sys.stdout = self.stream
        sys.stderr = self.stream



        # Connect the Stream's newText signal to the appendText method
        self.stream.newText.connect(self.appendText)

        self.button.clicked.connect(self.on_button_clicked)
        self.button_stop.clicked.connect(self.on_button_stop_clicked)
        self.button_data.clicked.connect(self.on_button_data_clicked)

        self.radio_outdata.setChecked(True)  # Set 'Option 1' as the default selected option

        self.worker = Worker()
        #self.setGeometry(300, 300, 300, 200)
        print("当前版本：v1.0.23.0626")
    def init_menuBar(self):
        # Create a menu bar
        menubar = self.menuBar()

        # Create a menu
        fileMenu = menubar.addMenu('File')
        viewMenu = menubar.addMenu('View')

        # Create an action
        openAction = QAction('Open', self)
        openAction.triggered.connect(self.openFile)  # Connect the triggered signal to the openFile slot
        fileMenu.addAction(openAction)

        # 查看曲线
        openAction1 = QAction('Tensorboard', self)
        openAction1.triggered.connect(self.on_button_tensorboard_clicked)  # Connect the triggered signal to the openFile slot
        viewMenu.addAction(openAction1)

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if fileName:
            print(f'File opened: {fileName}')

    def init_tab_creat_data(self):
        self.tab1 = QWidget()
        self.layout_creat_data = QVBoxLayout(self.tab1)

        # 数据集
        label = QLabel('数据集路径:')
        self.lineEdit_data = QLineEdit(self.tab1)
        self.button_data = QPushButton('选择数据集')
        self.layout_data = QHBoxLayout(self.tab1)
        self.layout_data.addWidget(label)
        self.layout_data.addWidget(self.lineEdit_data)
        self.layout_data.addWidget(self.button_data)
        self.layout_creat_data.addLayout(self.layout_data)

        self.layout_data = QHBoxLayout(self.tab1)
        label = QLabel('类别数量:')
        self.lineEdit_data_cnt = QLineEdit(self.tab1)
        self.lineEdit_data_cnt.setText('80')
        self.layout_data.addWidget(label)
        self.layout_data.addWidget(self.lineEdit_data_cnt)
        self.layout_creat_data.addLayout(self.layout_data)

        self.layout_data = QHBoxLayout(self.tab1)
        label = QLabel('切分数据集:   ')
        self.layout_data.addWidget(label)

        label = QLabel('训练集:')
        self.lineEdit_data_train = QLineEdit(self.tab1)
        self.layout_data.addWidget(label)
        self.layout_data.addWidget(self.lineEdit_data_train)
        label = QLabel('验证集:')
        self.lineEdit_data_val = QLineEdit(self.tab1)
        self.layout_data.addWidget(label)
        self.layout_data.addWidget(self.lineEdit_data_val)
        label = QLabel('测试集:')
        self.lineEdit_data_test = QLineEdit(self.tab1)
        self.layout_data.addWidget(label)
        self.layout_data.addWidget(self.lineEdit_data_test)
        self.layout_creat_data.addLayout(self.layout_data)
        self.layout_creat_data.addStretch(1)  # Add a stretchable space at the bottom
        self.tabWidget.addTab(self.tab1, "创建数据集")

    def init_tab_train(self):

        self.tab2 = QWidget()
        self.layout_train = QVBoxLayout(self.tab2)
        # 按钮
        self.button = QPushButton('开始训练')
        self.button_stop = QPushButton('结束训练')
        self.tensorboardFlg = False
        self.layout_bts = QHBoxLayout(self.tab2)
        self.layout_bts.addWidget(self.button)
        self.layout_bts.addWidget(self.button_stop)
        self.layout_train.addLayout(self.layout_bts)

        # 数据集
        self.radio_outdata = QRadioButton('外部数据集   ', self)
        label = QLabel('数据集路径:')
        label.setStyleSheet("color: gray;")
        # label.setStyleSheet("color: black; disabled { color: gray; }")
        self.lineEdit_data = QLineEdit(self.tab2)
        self.button_data = QPushButton('选择数据集')
        self.layout_data = QHBoxLayout(self.tab2)
        self.layout_data.addWidget(self.radio_outdata)
        self.layout_data.addWidget(label)
        self.layout_data.addWidget(self.lineEdit_data)
        self.layout_data.addWidget(self.button_data)
        self.layout_train.addLayout(self.layout_data)

        self.layout_data = QHBoxLayout(self.tab2)
        label = QLabel('类别数量:')
        self.lineEdit_data_cnt = QLineEdit(self.tab2)
        self.lineEdit_data_cnt.setText('80')
        self.layout_data.addWidget(label)
        self.layout_data.addWidget(self.lineEdit_data_cnt)
        self.layout_train.addLayout(self.layout_data)

        self.layout_data = QHBoxLayout(self.tab2)
        label = QLabel('切分数据集:   ')
        self.layout_data.addWidget(label)

        label = QLabel('训练集:')
        self.lineEdit_data_train = QLineEdit(self.tab2)
        self.lineEdit_data_train.setText('70')
        self.layout_data.addWidget(label)
        self.layout_data.addWidget(self.lineEdit_data_train)
        label = QLabel('验证集:')
        self.lineEdit_data_val = QLineEdit(self.tab2)
        self.lineEdit_data_val.setText('10')
        self.layout_data.addWidget(label)
        self.layout_data.addWidget(self.lineEdit_data_val)
        label = QLabel('测试集:')
        self.lineEdit_data_test = QLineEdit(self.tab2)
        self.lineEdit_data_test.setText('20')
        self.layout_data.addWidget(label)
        self.layout_data.addWidget(self.lineEdit_data_test)

        self.layout_train.addLayout(self.layout_data)

        # 迭代次数
        self.label = QLabel('迭代次数:')
        self.lineEdit = QLineEdit(self.tab2)
        self.layout_epochs = QHBoxLayout(self.tab2)
        self.layout_epochs.addWidget(self.label)
        self.layout_epochs.addWidget(self.lineEdit)
        self.layout_train.addLayout(self.layout_epochs)

        # imgsz
        self.layout_GPU = QHBoxLayout(self.tab2)
        label0 = QLabel('图片尺寸:')
        self.comboBox = QComboBox(self.tab2)
        self.comboBox.addItem("224")
        self.layout_GPU.addWidget(label0)
        self.layout_GPU.addWidget(self.comboBox)
        self.layout_train.addLayout(self.layout_GPU)

        self.layout_bar = QHBoxLayout(self.tab2)
        label1 = QLabel('训练进度:')
        self.progressBar = QProgressBar(self.tab2)
        self.layout_bar.addWidget(label1)
        self.layout_bar.addWidget(self.progressBar)
        self.layout_train.addLayout(self.layout_bar)
        # Create QTextEdit and layout
        self.textEdit = QTextEdit(self.tab2)

        # self.layout_train.addWidget(self.comboBox)
        # self.layout_train.addWidget(self.progressBar)

        self.layout_train.addWidget(self.textEdit)
        self.tabWidget.addTab(self.tab2, "训练")

        self.radio_outdata.toggled.connect(self.on_radio_button_toggled1)
        # self.radio2.toggled.connect(self.on_radio_button_toggled2)

    def init_tab_infrence(self):
        self.tab_infrence = QWidget()
        self.layout_infrence = QVBoxLayout(self.tab_infrence)
        self.tabWidget.addTab(self.tab_infrence, "导出模型")

    def init_tab_test(self):
        self.tab_test = QWidget()
        self.layout_test = QVBoxLayout(self.tab_test)
        self.tabWidget.addTab(self.tab_test, "测试")

    def appendText(self, text):
        text = text.replace('\r', '')
        self.textEdit.moveCursor(QTextCursor.End)
        self.textEdit.insertPlainText(text)

    def on_button_clicked(self):
        self.worker.start()

    def on_button_stop_clicked(self):
        self.worker.terminate()

    def on_button_data_clicked(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        print(folder_path)  # Print the selected folder path
        self.lineEdit_data.setText(folder_path)

    def on_radio_button_toggled1(self):
        pass

class mainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('KADO AI训练 v1.0.23.0626')
        self.resize(800, 600)
        self.init_menuBar()

        # Create a central widget
        self.centralWidget = MyWindow(self)
        self.setCentralWidget(self.centralWidget)

        self.tensorboardRun = TensorboardRun()
        self.show()

    def init_menuBar(self):
        # Create a menu bar
        menubar = self.menuBar()

        # Create a menu
        fileMenu = menubar.addMenu('文件')
        viewMenu = menubar.addMenu('查看')

        # Create an action
        openAction = QAction('打开', self)
        openAction.triggered.connect(self.openFile)  # Connect the triggered signal to the openFile slot
        fileMenu.addAction(openAction)

        # 查看曲线
        openAction1 = QAction('训练曲线', self)
        openAction1.triggered.connect(self.on_button_tensorboard_clicked)  # Connect the triggered signal to the openFile slot
        viewMenu.addAction(openAction1)
    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if fileName:
            print(f'File opened: {fileName}')
    def on_button_tensorboard_clicked(self):
        self.tensorboardRun.start()

        browser = webbrowser.get()
        browser.open('http://localhost:6006/')
if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QApplication([])
    window = mainWindow()
    app.exec_()

