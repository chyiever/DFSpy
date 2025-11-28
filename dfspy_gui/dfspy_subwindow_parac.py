# -*- coding: utf-8 -*-

"""
DFSPy GUI 参量转换子窗口模块

本模块实现应变到速度的参量转换功能。
使用 dfspy_cores.py 中的 strain_to_velocity 函数进行实际处理。
"""

import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog, QInputDialog
from PyQt5.QtCore import QThread, pyqtSignal

# 绘图需要
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

plt.rcParams['font.sans-serif'] = ['Times New Roman']


class ParacWorker(QThread):
    """参量转换工作线程"""
    finished = pyqtSignal(str)  # 转换完成信号
    error = pyqtSignal(str)  # 错误信号
    
    def __init__(self, input_path, apparent_velocity, normalize_divisor):
        super().__init__()
        self.input_path = input_path
        self.apparent_velocity = apparent_velocity
        self.normalize_divisor = normalize_divisor
    
    def run(self):
        """执行参量转换"""
        try:
            # 导入核心模块
            import sys
            sys.path.append('..')
            import dfspy_cores
            
            output_path = dfspy_cores.strain_to_velocity(
                self.input_path,
                self.apparent_velocity,
                self.normalize_divisor
            )
            
            self.finished.emit(output_path)
        except Exception as e:
            self.error.emit(f"Conversion failed: {str(e)}")


class Ui_SubParac(QMainWindow):
    """参量转换子窗口 UI 类"""

    def __init__(self):
        super(Ui_SubParac, self).__init__()
        self.setupUi(self)

    def setupUi(self, SubParac):
        """设置 UI"""
        SubParac.setObjectName("SubParac")
        SubParac.resize(1881, 903)
        SubParac.setStyleSheet("background:rgb(240, 240, 240)")
        
        # 操作标签
        self.label_head_oper = QtWidgets.QLabel(SubParac)
        self.label_head_oper.setGeometry(QtCore.QRect(0, 0, 621, 51))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(12)
        self.label_head_oper.setFont(font)
        self.label_head_oper.setStyleSheet("background-color: rgb(211, 211, 211);")
        self.label_head_oper.setAlignment(QtCore.Qt.AlignCenter)
        self.label_head_oper.setObjectName("label_head_oper")
        
        # 文件导入组框
        self.groupBox_open = QtWidgets.QGroupBox(SubParac)
        self.groupBox_open.setGeometry(QtCore.QRect(10, 70, 591, 271))
        self.groupBox_open.setStyleSheet("background-color: rgb(240, 240, 240);")
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(12)
        self.groupBox_open.setFont(font)
        self.groupBox_open.setObjectName("groupBox_open")
        
        # 按钮样式
        button_style = """QPushButton
{
border-radius: 10px;  
border: 0.5px groove gray;
border-style: outset;
background-color: rgb(255, 255, 255);
}
QPushButton:pressed
{
    padding-left:4px;
    padding-top:4px;
    background-color:rgb(230, 240, 255);
}"""
        
        # 数据导入按钮
        self.pushButton_data_in = QtWidgets.QPushButton(self.groupBox_open)
        self.pushButton_data_in.setGeometry(QtCore.QRect(20, 40, 170, 50))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.pushButton_data_in.setFont(font)
        self.pushButton_data_in.setStyleSheet(button_style)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/mainwindow/image/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_data_in.setIcon(icon)
        self.pushButton_data_in.setIconSize(QtCore.QSize(32, 32))
        self.pushButton_data_in.setObjectName("pushButton_data_in")
        
        # 波形显示按钮
        self.pushButton_plot_before = QtWidgets.QPushButton(self.groupBox_open)
        self.pushButton_plot_before.setGeometry(QtCore.QRect(400, 40, 170, 50))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.pushButton_plot_before.setFont(font)
        self.pushButton_plot_before.setStyleSheet(button_style)
        icon_plot = QtGui.QIcon()
        icon_plot.addPixmap(QtGui.QPixmap(":/mainwindow/image/plot.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_plot_before.setIcon(icon_plot)
        self.pushButton_plot_before.setIconSize(QtCore.QSize(32, 32))
        self.pushButton_plot_before.setObjectName("pushButton_plot_before")
        
        # 文件路径列表
        self.listWidget_datafile_path = QtWidgets.QListWidget(self.groupBox_open)
        self.listWidget_datafile_path.setGeometry(QtCore.QRect(20, 100, 551, 151))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.listWidget_datafile_path.setFont(font)
        self.listWidget_datafile_path.setStyleSheet(
            "background-color: rgb(255, 255, 255);"
            "border-radius: 5px;"
            "border: 0.5px rgb(220, 220, 220);"
        )
        self.listWidget_datafile_path.setObjectName("listWidget_datafile_path")
        
        # 参量转换操作组框
        self.groupBox_parac = QtWidgets.QGroupBox(SubParac)
        self.groupBox_parac.setGeometry(QtCore.QRect(10, 360, 591, 341))
        self.groupBox_parac.setStyleSheet("background-color: rgb(240, 240, 240);")
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(12)
        self.groupBox_parac.setFont(font)
        self.groupBox_parac.setObjectName("groupBox_parac")
        
        # 转换参数输入区域
        self.label_apparent_velocity = QtWidgets.QLabel(self.groupBox_parac)
        self.label_apparent_velocity.setGeometry(QtCore.QRect(20, 40, 200, 30))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.label_apparent_velocity.setFont(font)
        self.label_apparent_velocity.setObjectName("label_apparent_velocity")
        
        self.lineEdit_apparent_velocity = QtWidgets.QLineEdit(self.groupBox_parac)
        self.lineEdit_apparent_velocity.setGeometry(QtCore.QRect(230, 40, 150, 30))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(12)
        self.lineEdit_apparent_velocity.setFont(font)
        self.lineEdit_apparent_velocity.setText("3000.0")
        self.lineEdit_apparent_velocity.setStyleSheet(
            "border: 1px solid gray;"
            "border-radius: 5px;"
            "padding: 2px;"
            "background-color: rgb(255, 255, 255);"
        )
        self.lineEdit_apparent_velocity.setObjectName("lineEdit_apparent_velocity")
        
        self.label_normalize_divisor = QtWidgets.QLabel(self.groupBox_parac)
        self.label_normalize_divisor.setGeometry(QtCore.QRect(20, 80, 200, 30))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.label_normalize_divisor.setFont(font)
        self.label_normalize_divisor.setObjectName("label_normalize_divisor")
        
        self.lineEdit_normalize_divisor = QtWidgets.QLineEdit(self.groupBox_parac)
        self.lineEdit_normalize_divisor.setGeometry(QtCore.QRect(230, 80, 150, 30))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(12)
        self.lineEdit_normalize_divisor.setFont(font)
        self.lineEdit_normalize_divisor.setText("5000.0")
        self.lineEdit_normalize_divisor.setStyleSheet(
            "border: 1px solid gray;"
            "border-radius: 5px;"
            "padding: 2px;"
            "background-color: rgb(255, 255, 255);"
        )
        self.lineEdit_normalize_divisor.setObjectName("lineEdit_normalize_divisor")
        
        # 参数说明列表
        self.listWidget_params = QtWidgets.QListWidget(self.groupBox_parac)
        self.listWidget_params.setGeometry(QtCore.QRect(20, 120, 551, 141))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(9)
        self.listWidget_params.setFont(font)
        self.listWidget_params.setStyleSheet(
            "background-color: rgb(255, 255, 255);"
            "border-radius: 5px;"
            "border: 0.5px rgb(220, 220, 220);"
        )
        self.listWidget_params.setObjectName("listWidget_params")
        
        # Add parameter descriptions
        self.listWidget_params.addItem("Strain-Velocity Conversion Parameters:")
        self.listWidget_params.addItem("• Apparent velocity (m/s): seismic wave propagation speed on the surface, typically 3000-5000 m/s")
        self.listWidget_params.addItem("• Normalization divisor: divisor for data normalization, default 5000.0")
        self.listWidget_params.addItem("• Conversion formula: velocity = strain * apparent_velocity / normalize_divisor")
        
        # 开始转换按钮
        self.pushButton_begin = QtWidgets.QPushButton(self.groupBox_parac)
        self.pushButton_begin.setGeometry(QtCore.QRect(20, 280, 170, 50))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.pushButton_begin.setFont(font)
        self.pushButton_begin.setStyleSheet(button_style)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/mainwindow/image/图片3.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_begin.setIcon(icon2)
        self.pushButton_begin.setIconSize(QtCore.QSize(32, 32))
        self.pushButton_begin.setObjectName("pushButton_begin")
        
        # 转换后显示按钮
        self.pushButton_plot_after = QtWidgets.QPushButton(self.groupBox_parac)
        self.pushButton_plot_after.setGeometry(QtCore.QRect(400, 280, 170, 50))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.pushButton_plot_after.setFont(font)
        self.pushButton_plot_after.setStyleSheet(button_style)
        self.pushButton_plot_after.setIcon(icon_plot)
        self.pushButton_plot_after.setIconSize(QtCore.QSize(32, 32))
        self.pushButton_plot_after.setObjectName("pushButton_plot_after")
        
        # 转换前可视化组框
        self.groupBox_visu_before = QtWidgets.QGroupBox(SubParac)
        self.groupBox_visu_before.setGeometry(QtCore.QRect(630, 70, 620, 821))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(12)
        self.groupBox_visu_before.setFont(font)
        self.groupBox_visu_before.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.groupBox_visu_before.setObjectName("groupBox_visu_before")
        
        # 转换前绘图区域
        self.widget_plot_before = QtWidgets.QWidget(self.groupBox_visu_before)
        self.widget_plot_before.setGeometry(QtCore.QRect(10, 30, 601, 761))
        self.widget_plot_before.setObjectName("widget_plot_before")
        
        # 转换后可视化组框
        self.groupBox_visu_after = QtWidgets.QGroupBox(SubParac)
        self.groupBox_visu_after.setGeometry(QtCore.QRect(1260, 70, 620, 821))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(12)
        self.groupBox_visu_after.setFont(font)
        self.groupBox_visu_after.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.groupBox_visu_after.setObjectName("groupBox_visu_after")
        
        # 转换后绘图区域
        self.widget_plot_after = QtWidgets.QWidget(self.groupBox_visu_after)
        self.widget_plot_after.setGeometry(QtCore.QRect(10, 30, 601, 761))
        self.widget_plot_after.setObjectName("widget_plot_after")
        
        # 可视化标签
        self.label_head_plot = QtWidgets.QLabel(SubParac)
        self.label_head_plot.setGeometry(QtCore.QRect(620, 0, 1271, 51))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(16)
        self.label_head_plot.setFont(font)
        self.label_head_plot.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.label_head_plot.setAlignment(QtCore.Qt.AlignCenter)
        self.label_head_plot.setObjectName("label_head_plot")
        
        # 退出按钮
        self.pushButton_exit = QtWidgets.QPushButton(SubParac)
        self.pushButton_exit.setGeometry(QtCore.QRect(210, 840, 180, 50))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_exit.setFont(font)
        self.pushButton_exit.setStyleSheet(button_style)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/mainwindow/image/Exit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_exit.setIcon(icon3)
        self.pushButton_exit.setIconSize(QtCore.QSize(32, 32))
        self.pushButton_exit.setObjectName("pushButton_exit")

        self.retranslateUi(SubParac)
        self.pushButton_exit.clicked.connect(SubParac.close)
        QtCore.QMetaObject.connectSlotsByName(SubParac)

        # 连接信号和槽
        self.pushButton_data_in.clicked.connect(self.select_data_files)
        self.pushButton_plot_before.clicked.connect(self.plot_before)
        self.pushButton_plot_after.clicked.connect(self.plot_after)
        self.pushButton_begin.clicked.connect(self.start_conversion)

        # 初始化绘图
        self.setup_plotting()
        self.output_file_path = None  # 存储转换后的文件路径

    def setup_plotting(self):
        """设置绘图区域"""
        # 转换前绘图
        self.fig_before = plt.Figure()
        self.canvas_before = FC(self.fig_before)
        layout_before = QtWidgets.QVBoxLayout()
        layout_before.addWidget(self.canvas_before)
        toolbar_before = NavigationToolbar(self.canvas_before, self)
        layout_before.addWidget(toolbar_before)
        self.widget_plot_before.setLayout(layout_before)

        # 转换后绘图
        self.fig_after = plt.Figure()
        self.canvas_after = FC(self.fig_after)
        layout_after = QtWidgets.QVBoxLayout()
        layout_after.addWidget(self.canvas_after)
        toolbar_after = NavigationToolbar(self.canvas_after, self)
        layout_after.addWidget(toolbar_after)
        self.widget_plot_after.setLayout(layout_after)

    def select_data_files(self):
        """选择数据文件"""
        # 检查并设置默认路径
        import os
        default_path = ""
        if os.path.exists("../exampledata"):
            default_path = "../exampledata"
        
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select data files",
            default_path,
            "Seismic Data Files (*.mseed *.sac);;All Files (*.*)"
        )
        if files:
            self.listWidget_datafile_path.clear()
            self.listWidget_datafile_path.addItems(files)

    def plot_before(self):
        """绘制转换前数据"""
        if self.listWidget_datafile_path.count() == 0:
            QMessageBox.warning(self, "Warning", "Please select data files first!")
            return

        try:
            # 导入核心模块绘图函数
            import sys
            sys.path.append('..')
            import dfspy_cores

            file_path = self.listWidget_datafile_path.item(0).text()
            
            ax = self.fig_before.add_subplot(111)
            ax.clear()
            
            st = dfspy_cores.read_stream(file_path)
            dfspy_cores.plot_stream(st, "Data Before Conversion (Strain)", ax)
            
            self.canvas_before.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Plotting failed: {str(e)}")

    def plot_after(self):
        """绘制转换后数据"""
        if not self.output_file_path:
            QMessageBox.warning(self, "Warning", "Please perform the parameter conversion first!")
            return

        try:
            # 导入核心模块绘图函数
            import sys
            sys.path.append('..')
            import dfspy_cores

            ax = self.fig_after.add_subplot(111)
            ax.clear()
            
            st = dfspy_cores.read_stream(self.output_file_path)
            dfspy_cores.plot_stream(st, "Data After Conversion (Velocity)", ax)
            
            self.canvas_after.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Plotting failed: {str(e)}")

    def start_conversion(self):
        """开始参量转换"""
        if self.listWidget_datafile_path.count() == 0:
            QMessageBox.warning(self, "Warning", "Please select data files first!")
            return

        # 获取参数
        try:
            apparent_velocity = float(self.lineEdit_apparent_velocity.text())
            normalize_divisor = float(self.lineEdit_normalize_divisor.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid numeric parameters!")
            return

        if apparent_velocity <= 0 or normalize_divisor <= 0:
            QMessageBox.warning(self, "Warning", "Parameter values must be greater than 0!")
            return

        file_path = self.listWidget_datafile_path.item(0).text()
        
        # 开始转换
        self.worker = ParacWorker(file_path, apparent_velocity, normalize_divisor)
        self.worker.finished.connect(self.on_conversion_finished)
        self.worker.error.connect(self.on_conversion_error)
        
        self.pushButton_begin.setEnabled(False)
        self.pushButton_begin.setText("Converting...")
        self.worker.start()

    def on_conversion_finished(self, output_path):
        """转换完成回调"""
        self.pushButton_begin.setEnabled(True)
        self.pushButton_begin.setText("Start Conversion")
        self.output_file_path = output_path

        self.listWidget_params.addItem(f"Conversion finished! Output file: {output_path}")

        QMessageBox.information(self, "Success", "Parameter conversion completed!")

    def on_conversion_error(self, error_message):
        """转换错误回调"""
        self.pushButton_begin.setEnabled(True)
        self.pushButton_begin.setText("Start Conversion")
        QMessageBox.critical(self, "Error", error_message)

    def retranslateUi(self, SubParac):
        """设置 UI 文本"""
        _translate = QtCore.QCoreApplication.translate
        SubParac.setWindowTitle(_translate("SubParac", "Strain-Velocity Conversion"))
        self.label_head_oper.setText(_translate("SubParac", "Operations"))
        self.groupBox_open.setTitle(_translate("SubParac", "File Import"))
        self.pushButton_data_in.setText(_translate("SubParac", "Import Data"))
        self.pushButton_plot_before.setText(_translate("SubParac", "Plot Waveform"))
        self.groupBox_parac.setTitle(_translate("SubParac", "Strain-Velocity Conversion"))
        self.label_apparent_velocity.setText(_translate("SubParac", "Apparent Velocity (m/s):"))
        self.label_normalize_divisor.setText(_translate("SubParac", "Normalization Divisor:"))
        self.pushButton_begin.setText(_translate("SubParac", "Start Conversion"))
        self.pushButton_plot_after.setText(_translate("SubParac", "Plot After Conversion"))
        self.groupBox_visu_before.setTitle(_translate("SubParac", "Before Conversion"))
        self.groupBox_visu_after.setTitle(_translate("SubParac", "After Conversion"))
        self.label_head_plot.setText(_translate("SubParac", "Visualization"))
        self.pushButton_exit.setText(_translate("SubParac", "Exit"))


import imag_qrc_rc