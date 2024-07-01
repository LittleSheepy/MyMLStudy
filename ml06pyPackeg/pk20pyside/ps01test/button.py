from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
    QDialog, QGroupBox, QHeaderView, QLabel,
    QLineEdit, QListWidgetItem, QSizePolicy, QStackedWidget,
    QTextEdit, QToolButton, QWidget)
from PySide6.QtWidgets import (QTableWidget, QTableWidgetItem, QAbstractItemView,
                               QListWidget, QListWidgetItem,
                               QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                               QGraphicsPolygonItem, QGraphicsTextItem, QGraphicsItem, QAbstractGraphicsShapeItem)
from typing import List, Any, Union, Dict
# 单列显示控件
class MyQListWidget(QListWidget):
    def __init__(self, obj: QtCore.QObject):
        super().__init__(obj)

        self.setFont(QFont("Arial", 20))

    def UpdateListItem(self, listItem: List[str]):
        self.clear()
        for strText in listItem:
            item = QListWidgetItem()
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setText(strText)
            self.addItem(item)