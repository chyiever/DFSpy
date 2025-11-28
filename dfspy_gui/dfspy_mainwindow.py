# -*- coding: utf-8 -*-

"""
DFSPy GUI 主窗口模块

本模块实现 DFSPy 软件的主窗口界面，提供四个数据处理子窗口的入口：
- 数据压缩与重构
- 存储格式转换
- 数据降噪
- 应变-速度转换

使用 dfspy_cores.py 作为数据处理的底层功能模块。
"""

from PyQt5 import QtCore, QtGui, QtWidgets


class ClickableLabel(QtWidgets.QLabel):
    """可点击的标签类"""
    clicked = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.setCursor(QtCore.Qt.PointingHandCursor)  # 设置鼠标悬停时显示手型光标
    
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
        super(ClickableLabel, self).mousePressEvent(event)


class Ui_MainWindow(object):
    """DFSPy 主窗口 UI 类"""

    def setupUi(self, MainWindow):
        """设置主窗口 UI"""
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 900)
        MainWindow.setStyleSheet("background:rgb(255, 255, 255)")
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # 标题标签
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 20, 900, 91))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        
        # 按钮样式
        button_style = """QPushButton
{
border-radius: 10px;  
/* border: 0.5px groove gray;*/
border-style: outset;
background-color: rgb(255, 255, 255);
      
}
QPushButton:pressed
{
    
    /*按下时字向右移动4像素*/  
    padding-left:4px;
    /*按下时字向下移动4像素*/  
    padding-top:4px;
    background-color:rgb(230, 240, 255);
}"""
        
        # 数据压缩按钮
        self.pushButton_Compress = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Compress.setGeometry(QtCore.QRect(170, 390, 250, 61))
        self.pushButton_Compress.setMinimumSize(QtCore.QSize(131, 61))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_Compress.setFont(font)
        self.pushButton_Compress.setStyleSheet(button_style)
        self.pushButton_Compress.setAutoDefault(False)
        self.pushButton_Compress.setDefault(False)
        self.pushButton_Compress.setFlat(True)
        self.pushButton_Compress.setObjectName("pushButton_Compress")
        
        # 格式转换按钮
        self.pushButton_Formatc = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Formatc.setGeometry(QtCore.QRect(470, 390, 250, 61))
        self.pushButton_Formatc.setMinimumSize(QtCore.QSize(131, 61))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_Formatc.setFont(font)
        self.pushButton_Formatc.setStyleSheet(button_style)
        self.pushButton_Formatc.setAutoDefault(False)
        self.pushButton_Formatc.setDefault(False)
        self.pushButton_Formatc.setFlat(True)
        self.pushButton_Formatc.setObjectName("pushButton_Formatc")
        
        # 数据降噪按钮
        self.pushButton_Denoise = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Denoise.setGeometry(QtCore.QRect(170, 730, 250, 61))
        self.pushButton_Denoise.setMinimumSize(QtCore.QSize(131, 61))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_Denoise.setFont(font)
        self.pushButton_Denoise.setStyleSheet(button_style)
        self.pushButton_Denoise.setAutoDefault(False)
        self.pushButton_Denoise.setFlat(True)
        self.pushButton_Denoise.setObjectName("pushButton_Denoise")
        
        # 参量转换按钮
        self.pushButton_Parac = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Parac.setGeometry(QtCore.QRect(470, 730, 250, 61))
        self.pushButton_Parac.setMinimumSize(QtCore.QSize(131, 61))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_Parac.setFont(font)
        self.pushButton_Parac.setStyleSheet(button_style)
        self.pushButton_Parac.setCheckable(False)
        self.pushButton_Parac.setAutoDefault(False)
        self.pushButton_Parac.setDefault(False)
        self.pushButton_Parac.setFlat(True)
        self.pushButton_Parac.setObjectName("pushButton_Parac")
        
        # 图标标签 (可点击)
        self.label_2 = ClickableLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(180, 170, 221, 201))
        self.label_2.setStyleSheet("border-image: url(:/mainwindow/image/图片1.png);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        
        self.label_3 = ClickableLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(470, 160, 231, 221))
        self.label_3.setStyleSheet("border-image: url(:/mainwindow/image/图片2.png);")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        
        self.label_4 = ClickableLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(200, 540, 191, 181))
        self.label_4.setStyleSheet("border-image: url(:/mainwindow/image/图片4.png);")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        
        self.label_5 = ClickableLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(500, 530, 191, 191))
        self.label_5.setStyleSheet("border-image: url(:/mainwindow/image/图片3.png);")
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        # 菜单栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        
        # 状态栏
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        # 菜单项
        self.action_7 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(9)
        self.action_7.setFont(font)
        self.action_7.setObjectName("action_7")
        
        self.action_2 = QtWidgets.QAction(MainWindow)
        self.action_2.setObjectName("action_2")
        
        self.action_4 = QtWidgets.QAction(MainWindow)
        self.action_4.setObjectName("action_4")
        
        self.action_6 = QtWidgets.QAction(MainWindow)
        self.action_6.setObjectName("action_6")
        
        self.menu.addAction(self.action_7)
        self.menu_2.addAction(self.action_2)
        self.menu_2.addAction(self.action_4)
        self.menu_2.addAction(self.action_6)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        # 连接按钮信号到子窗口打开槽函数
        self.pushButton_Parac.clicked.connect(self.open_Parac)
        self.pushButton_Denoise.clicked.connect(self.open_Denoise)
        self.pushButton_Formatc.clicked.connect(self.open_Formatc)
        self.pushButton_Compress.clicked.connect(self.open_Compress)
        
        # 连接图标标签点击事件到子窗口打开槽函数
        self.label_2.clicked.connect(self.open_Compress)    # 图片1.png -> 压缩
        self.label_3.clicked.connect(self.open_Formatc)    # 图片2.png -> 格式转换
        self.label_4.clicked.connect(self.open_Denoise)    # 图片4.png -> 降噪
        self.label_5.clicked.connect(self.open_Parac)      # 图片3.png -> 参量转换
        
        # 连接菜单项信号
        self.action_4.triggered.connect(self.show_contact_info)  # 联系我们
        self.action_6.triggered.connect(self.show_about_info)    # 关于

    def open_Formatc(self):
        """打开格式转换子窗口"""
        import dfspy_subwindow_formatc
        self.subwindow1 = dfspy_subwindow_formatc.Ui_SubFormatc()
        self.subwindow1.show()
    
    def open_Compress(self):
        """打开数据压缩子窗口"""
        import dfspy_subwindow_compress
        self.subwindow2 = dfspy_subwindow_compress.Ui_SubCompress()
        self.subwindow2.show()
    
    def open_Denoise(self):
        """打开数据降噪子窗口"""
        import dfspy_subwindow_denoise
        self.subwindow3 = dfspy_subwindow_denoise.Ui_SubDenoise()
        self.subwindow3.show()
    
    def open_Parac(self):
        """打开参量转换子窗口"""
        import dfspy_subwindow_parac
        self.subwindow4 = dfspy_subwindow_parac.Ui_SubParac()
        self.subwindow4.show()

    def show_contact_info(self):
        """显示联系我们信息"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(
            None,
            "Contact Us",
            "Contact Email: qi.gh@outlook.com",
            QMessageBox.Ok
        )

    def show_about_info(self):
        """显示关于信息"""
        from PyQt5.QtWidgets import QMessageBox
        about_text = """Software Introduction: DFSPy is a professional tool for distributed fiber-optic sensing data processing, designed for researchers to provide efficient and reliable data-processing solutions. The software integrates a variety of advanced data-processing algorithms, supporting preprocessing, analysis, and visualization of data collected by distributed fiber-optic sensing systems, helping researchers extract value from data and accelerate research progress.

Usage Statement:
1. This software is an open-source research tool developed by the Institute of Semiconductors, Chinese Academy of Sciences. It is provided for academic research and non-commercial use only. Users must comply with applicable laws and research ethics and must not use the software for commercial or illegal activities.
2. The software is provided "as is". The developers make no express or implied warranties about its suitability, completeness, or accuracy.
3. If you cite this software in academic publications, please acknowledge the software source and the developers.
\nContact Email: qi.gh@outlook.com"""

        QMessageBox.information(
            None,
            "About DFSPy",
            about_text,
            QMessageBox.Ok
        )

    def retranslateUi(self, MainWindow):
        """设置 UI 文本"""
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DFSPy - Main Window"))
        self.pushButton_Compress.setText(_translate("MainWindow", "Compression"))
        self.pushButton_Formatc.setText(_translate("MainWindow", "Format Convert"))
        self.pushButton_Denoise.setText(_translate("MainWindow", "Denoising"))
        self.pushButton_Parac.setText(_translate("MainWindow", "Strain-Velocity"))
        self.label.setText(_translate("MainWindow", "DFSPy - Data Processing Suite"))
        self.menu.setTitle(_translate("MainWindow", "Settings"))
        self.menu_2.setTitle(_translate("MainWindow", "Help"))
        self.action_2.setText(_translate("MainWindow", "Getting Started"))
        self.action_4.setText(_translate("MainWindow", "Contact Us"))
        self.action_6.setText(_translate("MainWindow", "About"))
        self.action_7.setText(_translate("MainWindow", "Options"))


import imag_qrc_rc

'''
**************************************************************************************  
软  件  名  称 : 分布式光纤传感数据处理软件 [简称： DFSPy] V1.0  
著  作  权  人 : 中国科学院半导体研究所(软件著作权登记号: 2025SR0353448)  
联  系  邮  箱 : qi.gh@outlook.com (开发者主页: https://github.com/chyiever)   
开  发  语  言 : Python 3.9+  
软  件  简  介：DFSPy 是一款专为科研人员设计的分布式光纤传感数据处理专业工具，致力于提供高效、可靠的数据处理解决方案。该软件集成了多种先进的数据处理算法，支持分布式光纤传感系统采集数据的预处理、分析与可视化，助力科研人员深入挖掘数据价值，加速研究进程。  
使  用  声  明：
1. 本软件为中国科学院半导体研究所开发的开源科研工具，仅供学术研究与非商业用途,用户使用本软件时，须遵守国家相关法律法规及科研道德规范，不得用于任何商业活动或非法用途。  
2. 软件以现状提供，开发者不对其适用性、完整性或准确性做出任何明示或暗示的保证。  
3. 如需引用本软件进行学术成果发表，请注明软件来源及开发者信息。  
***************************************************************************************  
'''