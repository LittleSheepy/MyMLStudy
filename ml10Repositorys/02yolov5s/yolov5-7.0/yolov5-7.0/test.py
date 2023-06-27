from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QTabWidget
import sys


class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout(self)

        self.tabWidget = QTabWidget(self)

        # Create first tab
        self.tab1 = QWidget()
        self.vbox1 = QVBoxLayout(self.tab1)
        self.button1 = QPushButton("Button 1")
        self.button2 = QPushButton("Button 2")
        self.vbox1.addWidget(self.button1)
        self.vbox1.addWidget(self.button2)

        # Create second tab
        self.tab2 = QWidget()
        self.vbox2 = QVBoxLayout(self.tab2)
        self.button3 = QPushButton("Button 3")
        self.button4 = QPushButton("Button 4")
        self.vbox2.addWidget(self.button3)
        self.vbox2.addWidget(self.button4)

        self.tabWidget.addTab(self.tab1, "Tab 1")
        self.tabWidget.addTab(self.tab2, "Tab 2")

        vbox.addWidget(self.tabWidget)
        vbox.addStretch(1)  # Add a stretchable space at the bottom
        self.setLayout(vbox)

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('QTabWidget demo')
        self.show()


def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()