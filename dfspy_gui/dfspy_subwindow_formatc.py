# -*- coding: utf-8 -*-

"""
DFSPy GUI 格式转换子窗口模块

本模块实现数据格式转换功能，支持 txt、SAC、MSEED、SEG-Y 格式之间的相互转换。
使用 dfspy_cores.py 中的 convert_format 函数进行实际的格式转换操作。
"""

import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal


class FormatConvertWorker(QThread):
    """格式转换工作线程"""
    finished = pyqtSignal(str)  # 转换完成信号
    error = pyqtSignal(str)  # 错误信号
    
    def __init__(self, input_files, output_format, headfile_path=None):
        super().__init__()
        self.input_files = input_files
        self.output_format = output_format
        self.headfile_path = headfile_path
    
    def run(self):
        """执行格式转换"""
        try:
            # 导入核心模块
            import sys
            sys.path.append('..')
            import dfspy_cores
            
            for file_path in self.input_files:
                output_path = dfspy_cores.convert_format(
                    file_path, 
                    self.output_format, 
                    self.headfile_path
                )
            
            self.finished.emit("格式转换完成！")
        except Exception as e:
            self.error.emit(f"Format conversion failed: {str(e)}")


class Ui_SubFormatc(QMainWindow):
    """格式转换子窗口 UI 类"""

    def __init__(self):
        super(Ui_SubFormatc, self).__init__()
        self.setupUi(self)

    def setupUi(self, SubFormatc):
        """设置 UI"""
        SubFormatc.setObjectName("SubFormatc")
        SubFormatc.resize(610, 914)
        SubFormatc.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        SubFormatc.setStyleSheet("background:rgb(240, 240, 240)")
        
        # 操作标签
        self.label_head_oper = QtWidgets.QLabel(SubFormatc)
        self.label_head_oper.setGeometry(QtCore.QRect(0, 0, 611, 51))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(16)
        self.label_head_oper.setFont(font)
        self.label_head_oper.setStyleSheet("background-color: rgb(211, 211, 211);")
        self.label_head_oper.setAlignment(QtCore.Qt.AlignCenter)
        self.label_head_oper.setObjectName("label_head_oper")
        
        # 文件导入组框
        self.groupBox_open = QtWidgets.QGroupBox(SubFormatc)
        self.groupBox_open.setGeometry(QtCore.QRect(10, 70, 591, 451))
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
        
        # 数据文件导入按钮
        self.pushButton_data_in = QtWidgets.QPushButton(self.groupBox_open)
        self.pushButton_data_in.setGeometry(QtCore.QRect(20, 40, 211, 50))
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
        
        # 数据文件路径列表
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
        
        # 头文件导入按钮
        self.pushButton_headfile_in = QtWidgets.QPushButton(self.groupBox_open)
        self.pushButton_headfile_in.setGeometry(QtCore.QRect(20, 260, 241, 50))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.pushButton_headfile_in.setFont(font)
        self.pushButton_headfile_in.setStyleSheet(button_style)
        self.pushButton_headfile_in.setIcon(icon)
        self.pushButton_headfile_in.setIconSize(QtCore.QSize(32, 32))
        self.pushButton_headfile_in.setObjectName("pushButton_headfile_in")
        
        # 头文件路径列表
        self.listWidget_headfile_path = QtWidgets.QListWidget(self.groupBox_open)
        self.listWidget_headfile_path.setGeometry(QtCore.QRect(20, 320, 551, 111))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.listWidget_headfile_path.setFont(font)
        self.listWidget_headfile_path.setStyleSheet(
            "background-color: rgb(255, 255, 255);"
            "border-radius: 5px;"
            "border: 0.5px rgb(220, 220, 220);"
        )
        self.listWidget_headfile_path.setObjectName("listWidget_headfile_path")
        
        # 格式转换组框
        self.groupBox = QtWidgets.QGroupBox(SubFormatc)
        self.groupBox.setGeometry(QtCore.QRect(10, 540, 591, 231))
        self.groupBox.setStyleSheet("background-color: rgb(240, 240, 240);")
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        
        # 原始格式选择标签
        self.label_select_format0 = QtWidgets.QLabel(self.groupBox)
        self.label_select_format0.setGeometry(QtCore.QRect(20, 50, 311, 41))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.label_select_format0.setFont(font)
        self.label_select_format0.setObjectName("label_select_format0")
        
        # 原始格式下拉框
        self.comboBox_format_in = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_format_in.setGeometry(QtCore.QRect(370, 50, 171, 41))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.comboBox_format_in.setFont(font)
        self.comboBox_format_in.setStyleSheet(
            "border: 0.5px groove gray;"
            "border-style: outset;"
            "background-color: rgb(255, 255, 255);"
        )
        self.comboBox_format_in.addItem("txt")
        self.comboBox_format_in.addItem("SAC")
        self.comboBox_format_in.addItem("SEG-Y")
        self.comboBox_format_in.addItem("MSEED")
        self.comboBox_format_in.setObjectName("comboBox_format_in")
        
        # 目标格式选择标签
        self.label_select_format1 = QtWidgets.QLabel(self.groupBox)
        self.label_select_format1.setGeometry(QtCore.QRect(20, 100, 321, 41))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.label_select_format1.setFont(font)
        self.label_select_format1.setObjectName("label_select_format1")
        
        # 目标格式下拉框
        self.comboBox_format_out = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_format_out.setGeometry(QtCore.QRect(370, 100, 171, 41))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.comboBox_format_out.setFont(font)
        self.comboBox_format_out.setStyleSheet(
            "border: 0.5px groove gray;"
            "border-style: outset;"
            "background-color: rgb(255, 255, 255);"
        )
        self.comboBox_format_out.addItem("SAC")
        self.comboBox_format_out.addItem("SEG-Y")
        self.comboBox_format_out.addItem("MSEED")
        self.comboBox_format_out.addItem("txt")
        self.comboBox_format_out.setObjectName("comboBox_format_out")
        
        # 开始转换按钮
        self.pushButton_begin = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_begin.setGeometry(QtCore.QRect(20, 160, 241, 50))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.pushButton_begin.setFont(font)
        self.pushButton_begin.setStyleSheet(button_style)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/mainwindow/image/图片2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_begin.setIcon(icon1)
        self.pushButton_begin.setIconSize(QtCore.QSize(35, 35))
        self.pushButton_begin.setObjectName("pushButton_begin")
        
        # 退出按钮
        self.pushButton_exit = QtWidgets.QPushButton(SubFormatc)
        self.pushButton_exit.setGeometry(QtCore.QRect(200, 850, 221, 50))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_exit.setFont(font)
        self.pushButton_exit.setStyleSheet(button_style)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/mainwindow/image/Exit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_exit.setIcon(icon2)
        self.pushButton_exit.setIconSize(QtCore.QSize(32, 32))
        self.pushButton_exit.setObjectName("pushButton_exit")

        self.retranslateUi(SubFormatc)
        self.pushButton_exit.clicked.connect(SubFormatc.close)
        QtCore.QMetaObject.connectSlotsByName(SubFormatc)

        # 连接信号和槽
        self.pushButton_data_in.clicked.connect(self.select_data_files)
        self.pushButton_headfile_in.clicked.connect(self.select_head_file)
        self.pushButton_begin.clicked.connect(self.start_format_conversion)

    def select_data_files(self):
        """选择数据文件"""
        # 设置默认路径为exampledata，如果不存在则使用当前目录
        import os
        default_path = "../exampledata" if os.path.exists("../exampledata") else ""
        
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select data files",
            default_path,
            "Seismic Data Files (*.txt *.sac *.mseed *.segy);;All Files (*.*)"
        )
        if files:
            self.listWidget_datafile_path.clear()
            self.listWidget_datafile_path.addItems(files)

    def select_head_file(self):
        """选择头文件"""
        # 设置默认路径为exampledata，如果不存在则使用当前目录
        import os
        default_path = "../exampledata" if os.path.exists("../exampledata") else ""
        
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select head file",
            default_path,
            "Header files (*.txt);;All Files (*.*)"
        )
        if file:
            self.listWidget_headfile_path.clear()
            self.listWidget_headfile_path.addItem(file)

    def start_format_conversion(self):
        """开始格式转换"""
        # 获取输入文件列表
        input_files = []
        for i in range(self.listWidget_datafile_path.count()):
            input_files.append(self.listWidget_datafile_path.item(i).text())
        
        if not input_files:
            QMessageBox.warning(self, "Warning", "Please select files to convert first!")
            return
        
        # 获取头文件路径（如果有）
        headfile_path = None
        if self.listWidget_headfile_path.count() > 0:
            headfile_path = self.listWidget_headfile_path.item(0).text()
        
        # 获取输出格式
        output_format = self.comboBox_format_out.currentText()
        input_format = self.comboBox_format_in.currentText()
        
        # 如果是从txt转换，检查是否需要头文件
        if input_format == "txt" and output_format in ["SAC", "MSEED"] and not headfile_path:
            QMessageBox.warning(self, "Warning", "Converting from txt to SAC or MSEED requires a head file!")
            return
        
        # 创建工作线程
        self.worker = FormatConvertWorker(input_files, output_format, headfile_path)
        self.worker.finished.connect(self.on_conversion_finished)
        self.worker.error.connect(self.on_conversion_error)
        
        # 禁用开始按钮，防止重复点击
        self.pushButton_begin.setEnabled(False)
        self.pushButton_begin.setText("Converting...")
        
        # 启动转换
        self.worker.start()

    def on_conversion_finished(self, message):
        """转换完成回调"""
        self.pushButton_begin.setEnabled(True)
        self.pushButton_begin.setText("Start Conversion")
        QMessageBox.information(self, "Success", message)

    def on_conversion_error(self, error_message):
        """转换错误回调"""
        self.pushButton_begin.setEnabled(True)
        self.pushButton_begin.setText("Start Conversion")
        QMessageBox.critical(self, "Error", error_message)

    def retranslateUi(self, SubFormatc):
        """设置 UI 文本"""
        _translate = QtCore.QCoreApplication.translate
        SubFormatc.setWindowTitle(_translate("SubFormatc", "Format Conversion"))
        self.groupBox_open.setTitle(_translate("SubFormatc", "File Import"))
        self.pushButton_headfile_in.setText(_translate("SubFormatc", "Head File (optional)"))
        self.pushButton_data_in.setText(_translate("SubFormatc", "Import Data"))
        self.label_head_oper.setText(_translate("SubFormatc", "Operations"))
        self.groupBox.setTitle(_translate("SubFormatc", "Format Conversion"))
        self.pushButton_begin.setText(_translate("SubFormatc", "Start Conversion"))
        self.label_select_format0.setText(_translate("SubFormatc", "Select input format:"))
        self.label_select_format1.setText(_translate("SubFormatc", "Select output format:"))
        self.pushButton_exit.setText(_translate("SubFormatc", "Exit"))


import imag_qrc_rc