# -*- coding: utf-8 -*-

"""
DFSPy GUI 数据压缩子窗口模块

本模块实现数据压缩与重构功能，支持小波变换压缩。
使用 dfspy_cores.py 中的压缩相关函数进行实际处理。
"""

import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal

# 绘图需要
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

plt.rcParams['font.sans-serif'] = ['Times New Roman']


class CompressWorker(QThread):
    """压缩工作线程"""
    finished = pyqtSignal(str, float)  # 压缩完成信号 (output_path, compression_ratio)
    error = pyqtSignal(str)  # 错误信号
    
    def __init__(self, txt_path, method='fwt'):
        super().__init__()
        self.txt_path = txt_path
        self.method = method
    
    def run(self):
        """执行压缩"""
        try:
            # 导入核心模块
            import sys
            sys.path.append('..')
            import dfspy_cores
            
            if self.method == 'fwt':
                output_path, compression_ratio = dfspy_cores.compress_fwt_txt(self.txt_path)
                self.finished.emit(output_path, compression_ratio)
            else:
                self.error.emit("Unsupported compression method")
        except Exception as e:
            self.error.emit(f"Compression failed: {str(e)}")


class DecompressWorker(QThread):
    """解压工作线程"""
    finished = pyqtSignal(str)  # 解压完成信号
    error = pyqtSignal(str)  # 错误信号
    
    def __init__(self, pkl_path):
        super().__init__()
        self.pkl_path = pkl_path
    
    def run(self):
        """执行解压"""
        try:
            # 导入核心模块
            import sys
            sys.path.append('..')
            import dfspy_cores
            
            output_path = dfspy_cores.decompress_fwt_to_txt(self.pkl_path)
            self.finished.emit(output_path)
        except Exception as e:
            self.error.emit(f"Decompression failed: {str(e)}")


class Ui_SubCompress(QMainWindow):
    """数据压缩子窗口 UI 类"""

    def __init__(self):
        super(Ui_SubCompress, self).__init__()
        self.setupUi(self)

    def setupUi(self, SubCompress):
        """设置 UI"""
        SubCompress.setObjectName("SubCompress")
        SubCompress.resize(1861, 900)
        SubCompress.setStyleSheet("background:rgb(240, 240, 240)")
        
        # 操作标签
        self.label_head_oper = QtWidgets.QLabel(SubCompress)
        self.label_head_oper.setGeometry(QtCore.QRect(0, 0, 591, 51))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(16)
        self.label_head_oper.setFont(font)
        self.label_head_oper.setStyleSheet("background-color: rgb(211, 211, 211);")
        self.label_head_oper.setAlignment(QtCore.Qt.AlignCenter)
        self.label_head_oper.setObjectName("label_head_oper")
        
        # 文件导入组框
        self.groupBox_open = QtWidgets.QGroupBox(SubCompress)
        self.groupBox_open.setGeometry(QtCore.QRect(10, 70, 571, 331))
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
        self.pushButton_data_in.setGeometry(QtCore.QRect(20, 50, 180, 50))
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
        self.pushButton_plot_before.setGeometry(QtCore.QRect(380, 50, 170, 50))
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
        self.listWidget_datafile_path.setGeometry(QtCore.QRect(20, 110, 531, 201))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(9)
        self.listWidget_datafile_path.setFont(font)
        self.listWidget_datafile_path.setStyleSheet(
            "background-color: rgb(255, 255, 255);"
            "border-radius: 5px;"
            "border: 0.5px rgb(220, 220, 220);"
        )
        self.listWidget_datafile_path.setObjectName("listWidget_datafile_path")
        
        # 操作组框
        self.groupBox_opera = QtWidgets.QGroupBox(SubCompress)
        self.groupBox_opera.setGeometry(QtCore.QRect(10, 420, 571, 341))
        self.groupBox_opera.setStyleSheet("background-color: rgb(240, 240, 240);")
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(12)
        self.groupBox_opera.setFont(font)
        self.groupBox_opera.setObjectName("groupBox_opera")
        
        # 算法选择标签
        self.label_select_algorithm = QtWidgets.QLabel(self.groupBox_opera)
        self.label_select_algorithm.setGeometry(QtCore.QRect(20, 40, 261, 51))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.label_select_algorithm.setFont(font)
        self.label_select_algorithm.setObjectName("label_select_algorithm")
        
        # 算法选择下拉框
        self.comboBox_algorithm = QtWidgets.QComboBox(self.groupBox_opera)
        self.comboBox_algorithm.setGeometry(QtCore.QRect(270, 40, 281, 51))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.comboBox_algorithm.setFont(font)
        self.comboBox_algorithm.setStyleSheet(
            "border: 0.5px groove gray;"
            "border-style: outset;"
            "background-color: rgb(255, 255, 255);"
        )
        self.comboBox_algorithm.addItem("")
        self.comboBox_algorithm.addItem("FWT Compression")
        self.comboBox_algorithm.addItem("FWT Reconstruction")
        self.comboBox_algorithm.setObjectName("comboBox_algorithm")
        
        # 算法参数列表
        self.listWidget_algorithm_para = QtWidgets.QListWidget(self.groupBox_opera)
        self.listWidget_algorithm_para.setGeometry(QtCore.QRect(20, 100, 531, 161))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(9)
        self.listWidget_algorithm_para.setFont(font)
        self.listWidget_algorithm_para.setStyleSheet(
            "background-color: rgb(255, 255, 255);"
            "border-radius: 5px;"
            "border: 0.5px rgb(220, 220, 220);"
        )
        self.listWidget_algorithm_para.setObjectName("listWidget_algorithm_para")
        
        # 开始压缩按钮
        self.pushButton_begin = QtWidgets.QPushButton(self.groupBox_opera)
        self.pushButton_begin.setGeometry(QtCore.QRect(20, 270, 170, 50))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.pushButton_begin.setFont(font)
        self.pushButton_begin.setStyleSheet(button_style)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/mainwindow/image/图片1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_begin.setIcon(icon2)
        self.pushButton_begin.setIconSize(QtCore.QSize(32, 32))
        self.pushButton_begin.setObjectName("pushButton_begin")
        
        # 压缩前可视化组框
        self.groupBox_visu_before = QtWidgets.QGroupBox(SubCompress)
        self.groupBox_visu_before.setGeometry(QtCore.QRect(600, 70, 620, 821))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(12)
        self.groupBox_visu_before.setFont(font)
        self.groupBox_visu_before.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.groupBox_visu_before.setObjectName("groupBox_visu_before")
        
        # 压缩前绘图区域
        self.widget_plot_before = QtWidgets.QWidget(self.groupBox_visu_before)
        self.widget_plot_before.setGeometry(QtCore.QRect(10, 30, 601, 761))
        self.widget_plot_before.setObjectName("widget_plot_before")
        
        # 压缩后可视化组框
        self.groupBox_visu_after = QtWidgets.QGroupBox(SubCompress)
        self.groupBox_visu_after.setGeometry(QtCore.QRect(1230, 70, 620, 821))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(12)
        self.groupBox_visu_after.setFont(font)
        self.groupBox_visu_after.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.groupBox_visu_after.setObjectName("groupBox_visu_after")
        
        # 压缩后绘图区域
        self.widget_plot_after = QtWidgets.QWidget(self.groupBox_visu_after)
        self.widget_plot_after.setGeometry(QtCore.QRect(10, 30, 601, 761))
        self.widget_plot_after.setObjectName("widget_plot_after")
        
        # 可视化标签
        self.label_head_plot = QtWidgets.QLabel(SubCompress)
        self.label_head_plot.setGeometry(QtCore.QRect(590, 0, 1271, 51))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(16)
        self.label_head_plot.setFont(font)
        self.label_head_plot.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.label_head_plot.setAlignment(QtCore.Qt.AlignCenter)
        self.label_head_plot.setObjectName("label_head_plot")
        
        # 退出按钮
        self.pushButton_exit = QtWidgets.QPushButton(SubCompress)
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

        self.retranslateUi(SubCompress)
        self.pushButton_exit.clicked.connect(SubCompress.close)
        QtCore.QMetaObject.connectSlotsByName(SubCompress)

        # 连接信号和槽
        self.pushButton_data_in.clicked.connect(self.select_data_files)
        self.pushButton_plot_before.clicked.connect(self.plot_before)
        self.pushButton_begin.clicked.connect(self.start_compression)
        self.comboBox_algorithm.currentTextChanged.connect(self.show_algorithm_info)

        # 初始化绘图
        self.setup_plotting()

    def setup_plotting(self):
        """设置绘图区域"""
        # 压缩前绘图
        self.fig_before = plt.Figure()
        self.canvas_before = FC(self.fig_before)
        layout_before = QtWidgets.QVBoxLayout()
        layout_before.addWidget(self.canvas_before)
        toolbar_before = NavigationToolbar(self.canvas_before, self)
        layout_before.addWidget(toolbar_before)
        self.widget_plot_before.setLayout(layout_before)

        # 压缩后绘图
        self.fig_after = plt.Figure()
        self.canvas_after = FC(self.fig_after)
        layout_after = QtWidgets.QVBoxLayout()
        layout_after.addWidget(self.canvas_after)
        toolbar_after = NavigationToolbar(self.canvas_after, self)
        layout_after.addWidget(toolbar_after)
        self.widget_plot_after.setLayout(layout_after)

    def select_data_files(self):
        """选择数据文件"""
        # 设置默认路径为exampledata，如果不存在则使用当前目录
        import os
        default_path = "../exampledata" if os.path.exists("../exampledata") else ""
        
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select data files",
            default_path,
            "All Files (*.*);;Text Files (*.txt);;Seismic Data Files (*.mseed *.sac)"
        )
        if files:
            self.listWidget_datafile_path.clear()
            self.listWidget_datafile_path.addItems(files)

    def plot_before(self):
        """绘制压缩前数据"""
        if self.listWidget_datafile_path.count() == 0:
            QMessageBox.warning(self, "Warning", "Please select data files first!")
            return

        try:
            # 导入核心模块绘图函数
            import sys
            sys.path.append('..')
            import dfspy_cores

            file_path = self.listWidget_datafile_path.item(0).text()
            data = dfspy_cores.read_txt_array(file_path)
            
            # 先清空图框
            self.fig_before.clear()
            ax = self.fig_before.add_subplot(111)
            dfspy_cores.plot_array(data, "Data Before Compression", ax)
            self.canvas_before.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Plotting failed: {str(e)}")

    def show_algorithm_info(self):
        """显示算法信息"""
        algorithm = self.comboBox_algorithm.currentText()
        self.listWidget_algorithm_para.clear()
        
        if algorithm == "FWT Compression":
            self.listWidget_algorithm_para.addItem("FWT compression parameters: haar wavelet, zero padding, levels=2")
            self.pushButton_begin.setText("Start Compression")
        elif algorithm == "FWT Reconstruction":
            self.listWidget_algorithm_para.addItem("FWT reconstruction: rebuild original data from compressed file")
            self.pushButton_begin.setText("Start Reconstruction")

    def start_compression(self):
        """开始压缩/解压"""
        if self.listWidget_datafile_path.count() == 0:
            QMessageBox.warning(self, "Warning", "Please select data files first!")
            return

        algorithm = self.comboBox_algorithm.currentText()
        if not algorithm:
            QMessageBox.warning(self, "Warning", "Please select a compression algorithm!")
            return

        file_path = self.listWidget_datafile_path.item(0).text()
        
        if algorithm == "FWT Compression":
            if not file_path.endswith('.txt'):
                QMessageBox.warning(self, "Warning", "FWT compression only supports txt files!")
                return
            self.start_fwt_compression(file_path)
        elif algorithm == "FWT Reconstruction":
            if not file_path.endswith('.pkl'):
                QMessageBox.warning(self, "Warning", "FWT reconstruction requires selecting a pkl compressed file!")
                return
            self.start_fwt_decompression(file_path)

    def start_fwt_compression(self, file_path):
        """开始FWT压缩"""
        self.worker = CompressWorker(file_path, 'fwt')
        self.worker.finished.connect(self.on_compression_finished)
        self.worker.error.connect(self.on_operation_error)
        
        self.pushButton_begin.setEnabled(False)
        self.pushButton_begin.setText("Compressing...")
        self.worker.start()

    def start_fwt_decompression(self, file_path):
        """开始FWT解压"""
        self.worker = DecompressWorker(file_path)
        self.worker.finished.connect(self.on_decompression_finished)
        self.worker.error.connect(self.on_operation_error)
        
        self.pushButton_begin.setEnabled(False)
        self.pushButton_begin.setText("Decompressing...")
        self.worker.start()

    def on_compression_finished(self, output_path, compression_ratio):
        """压缩完成回调"""
        self.pushButton_begin.setEnabled(True)
        self.pushButton_begin.setText("Start Compression")

        self.listWidget_algorithm_para.addItem(f"Compression finished! Output file: {output_path}")
        self.listWidget_algorithm_para.addItem(f"Compression ratio: {compression_ratio:.2f}%")

        QMessageBox.information(self, "Success", "Compression completed!")

    def on_decompression_finished(self, output_path):
        """解压完成回调"""
        self.pushButton_begin.setEnabled(True)
        self.pushButton_begin.setText("Start Reconstruction")

        self.listWidget_algorithm_para.addItem(f"Reconstruction finished! Output file: {output_path}")
        
        # 尝试绘制重构后的数据
        try:
            import sys
            sys.path.append('..')
            import dfspy_cores
            
            data = dfspy_cores.read_txt_array(output_path)
            # 先清空图框
            self.fig_after.clear()
            ax = self.fig_after.add_subplot(111)
            dfspy_cores.plot_array(data, "Data After Reconstruction", ax)
            self.canvas_after.draw()
        except Exception as e:
            print(f"Failed to plot reconstructed data: {str(e)}")
        
        QMessageBox.information(self, "Success", "Reconstruction completed!")

    def on_operation_error(self, error_message):
        """操作错误回调"""
        self.pushButton_begin.setEnabled(True)
        self.pushButton_begin.setText("Start Processing")
        QMessageBox.critical(self, "Error", error_message)

    def retranslateUi(self, SubCompress):
        """设置 UI 文本"""
        _translate = QtCore.QCoreApplication.translate
        SubCompress.setWindowTitle(_translate("SubCompress", "Data Compression"))
        self.groupBox_visu_before.setTitle(_translate("SubCompress", "Before Compression"))
        self.groupBox_open.setTitle(_translate("SubCompress", "File Import"))
        self.pushButton_plot_before.setText(_translate("SubCompress", "Plot Waveform"))
        self.pushButton_data_in.setText(_translate("SubCompress", "Import Data"))
        self.label_head_oper.setText(_translate("SubCompress", "Operations"))
        self.groupBox_opera.setTitle(_translate("SubCompress", "Compression & Reconstruction"))
        self.pushButton_begin.setText(_translate("SubCompress", "Start Compression"))
        self.label_select_algorithm.setText(_translate("SubCompress", "Select operation:"))
        self.groupBox_visu_after.setTitle(_translate("SubCompress", "After Compression"))
        self.label_head_plot.setText(_translate("SubCompress", "Visualization"))
        self.pushButton_exit.setText(_translate("SubCompress", "Exit"))


import imag_qrc_rc