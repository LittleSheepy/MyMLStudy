from PyQt5.QtWidgets import QApplication, QWidget, QRadioButton, QVBoxLayout
import sys

class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        self.radio1 = QRadioButton('Option 1', self)
        self.radio2 = QRadioButton('Option 2', self)
        self.radio3 = QRadioButton('Option 3', self)

        self.radio1.setChecked(True)  # Set 'Option 1' as the default selected option

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.radio1)
        self.layout.addWidget(self.radio2)
        self.layout.addWidget(self.radio3)

        self.radio1.toggled.connect(self.on_radio_button_toggled1)
        self.radio2.toggled.connect(self.on_radio_button_toggled2)
        self.radio3.toggled.connect(self.on_radio_button_toggled3)

    def on_radio_button_toggled(self):
        self.radio1.setStyleSheet("color: gray;")
        self.radio2.setStyleSheet("color: gray;")
        self.radio3.setStyleSheet("color: gray;")
        if self.radio1.isChecked():
            self.radio1.setStyleSheet("color: black;")
        elif self.radio2.isChecked():
            self.radio2.setStyleSheet("color: black;")
        elif self.radio3.isChecked():
            self.radio3.setStyleSheet("color: black;")

    def on_radio_button_toggled1(self):
        print('Option 1 is selected.')
        self.on_radio_button_toggled()

    def on_radio_button_toggled2(self):
        print('Option 2 is selected.')
        self.on_radio_button_toggled()

    def on_radio_button_toggled3(self):
        print('Option 3 is selected.')
        self.on_radio_button_toggled()

app = QApplication(sys.argv)
window = MyApp()
window.show()
sys.exit(app.exec_())