# -*- coding: utf-8 -*-

"""
DFSPy GUI 数据降噪子窗口模块

本模块实现数据降噪功能，支持带通滤波、小波降噪、相关降噪、谱减法降噪等。
使用 dfspy_cores.py 中的降噪相关函数进行实际处理。
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


class DenoiseWorker(QThread):
    """降噪工作线程"""
    finished = pyqtSignal(str)  # 降噪完成信号
    error = pyqtSignal(str)  # 错误信号
    
    def __init__(self, input_path, method, params):
        super().__init__()
        self.input_path = input_path
        self.method = method
        self.params = params
    
    def run(self):
        """执行降噪"""
        try:
            # 导入核心模块
            import sys
            sys.path.append('..')
            import dfspy_cores
            
            if self.method == 'bandpass':
                output_path = dfspy_cores.bandpass_denoise(
                    self.input_path, 
                    self.params['freqmin'], 
                    self.params['freqmax']
                )
            elif self.method == 'correlation':
                output_path = dfspy_cores.correlation_denoise(
                    self.input_path,
                    window_size=self.params['window_size'],
                    step_size=self.params['step_size'],
                    corr_threshold=self.params['corr_threshold']
                )
            elif self.method == 'spectral_subtraction':
                output_path = dfspy_cores.spectral_subtraction_denoise(
                    self.input_path,
                    noise_frames=self.params['noise_frames'],
                    alpha=self.params['alpha'],
                    beta=self.params['beta'],
                    window=self.params['window'],
                    frame_length=self.params['frame_length'],
                    hop_length=self.params['hop_length']
                )
            elif self.method == 'wavelet':
                output_path = dfspy_cores.advanced_denoise(
                    self.input_path,
                    method='wavelet',
                    wavelet=self.params['wavelet'],
                    level=self.params['level']
                )
            else:
                raise ValueError(f"Unsupported denoise method: {self.method}")
            
            self.finished.emit(output_path)
        except Exception as e:
            self.error.emit(f"Denoising failed: {str(e)}")


class Ui_SubDenoise(QMainWindow):
    """数据降噪子窗口 UI 类"""

    def __init__(self):
        super(Ui_SubDenoise, self).__init__()
        self.setupUi(self)

    def setupUi(self, SubDenoise):
        """设置 UI"""
        SubDenoise.setObjectName("SubDenoise")
        SubDenoise.resize(1890, 902)
        SubDenoise.setStyleSheet("background:rgb(240, 240, 240)")
        
        # 操作标签
        self.label_head_oper = QtWidgets.QLabel(SubDenoise)
        self.label_head_oper.setGeometry(QtCore.QRect(0, 0, 621, 51))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(12)
        self.label_head_oper.setFont(font)
        self.label_head_oper.setStyleSheet("background-color: rgb(211, 211, 211);")
        self.label_head_oper.setAlignment(QtCore.Qt.AlignCenter)
        self.label_head_oper.setObjectName("label_head_oper")
        
        # 文件导入组框
        self.groupBox_open = QtWidgets.QGroupBox(SubDenoise)
        self.groupBox_open.setGeometry(QtCore.QRect(10, 70, 591, 331))
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
        self.pushButton_data_in.setGeometry(QtCore.QRect(20, 50, 170, 50))
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
        self.pushButton_plot_before.setGeometry(QtCore.QRect(400, 50, 170, 50))
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
        self.listWidget_datafile_path.setGeometry(QtCore.QRect(20, 110, 551, 201))
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
        
        # 降噪操作组框
        self.groupBox_denoise = QtWidgets.QGroupBox(SubDenoise)
        self.groupBox_denoise.setGeometry(QtCore.QRect(10, 420, 591, 341))
        self.groupBox_denoise.setStyleSheet("background-color: rgb(240, 240, 240);")
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(12)
        self.groupBox_denoise.setFont(font)
        self.groupBox_denoise.setObjectName("groupBox_denoise")
        
        # 降噪方法选择标签
        self.label_select_method = QtWidgets.QLabel(self.groupBox_denoise)
        self.label_select_method.setGeometry(QtCore.QRect(20, 40, 261, 51))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.label_select_method.setFont(font)
        self.label_select_method.setObjectName("label_select_method")
        
        # 降噪方法下拉框
        self.comboBox_method = QtWidgets.QComboBox(self.groupBox_denoise)
        self.comboBox_method.setGeometry(QtCore.QRect(290, 40, 281, 51))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.comboBox_method.setFont(font)
        self.comboBox_method.setStyleSheet(
            "border: 0.5px groove gray;"
            "border-style: outset;"
            "background-color: rgb(255, 255, 255);"
        )
        self.comboBox_method.addItem("")
        self.comboBox_method.addItem("Bandpass")
        self.comboBox_method.addItem("Wavelet Denoise")
        self.comboBox_method.addItem("Correlation Denoise")
        self.comboBox_method.addItem("Spectral Subtraction")
        self.comboBox_method.setObjectName("comboBox_method")
        
        # 参数信息列表
        self.listWidget_params = QtWidgets.QListWidget(self.groupBox_denoise)
        self.listWidget_params.setGeometry(QtCore.QRect(20, 100, 551, 161))
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
        
        # 开始降噪按钮
        self.pushButton_begin = QtWidgets.QPushButton(self.groupBox_denoise)
        self.pushButton_begin.setGeometry(QtCore.QRect(20, 270, 170, 50))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.pushButton_begin.setFont(font)
        self.pushButton_begin.setStyleSheet(button_style)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/mainwindow/image/图片4.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_begin.setIcon(icon2)
        self.pushButton_begin.setIconSize(QtCore.QSize(32, 32))
        self.pushButton_begin.setObjectName("pushButton_begin")
        
        # 降噪后显示按钮
        self.pushButton_plot_after = QtWidgets.QPushButton(self.groupBox_denoise)
        self.pushButton_plot_after.setGeometry(QtCore.QRect(400, 270, 170, 50))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(14)
        self.pushButton_plot_after.setFont(font)
        self.pushButton_plot_after.setStyleSheet(button_style)
        self.pushButton_plot_after.setIcon(icon_plot)
        self.pushButton_plot_after.setIconSize(QtCore.QSize(32, 32))
        self.pushButton_plot_after.setObjectName("pushButton_plot_after")
        
        # 降噪前可视化组框
        self.groupBox_visu_before = QtWidgets.QGroupBox(SubDenoise)
        self.groupBox_visu_before.setGeometry(QtCore.QRect(630, 70, 620, 821))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(12)
        self.groupBox_visu_before.setFont(font)
        self.groupBox_visu_before.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.groupBox_visu_before.setObjectName("groupBox_visu_before")
        
        # 降噪前绘图区域
        self.widget_plot_before = QtWidgets.QWidget(self.groupBox_visu_before)
        self.widget_plot_before.setGeometry(QtCore.QRect(10, 30, 601, 761))
        self.widget_plot_before.setObjectName("widget_plot_before")
        
        # 降噪后可视化组框
        self.groupBox_visu_after = QtWidgets.QGroupBox(SubDenoise)
        self.groupBox_visu_after.setGeometry(QtCore.QRect(1260, 70, 620, 821))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(12)
        self.groupBox_visu_after.setFont(font)
        self.groupBox_visu_after.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.groupBox_visu_after.setObjectName("groupBox_visu_after")
        
        # 降噪后绘图区域
        self.widget_plot_after = QtWidgets.QWidget(self.groupBox_visu_after)
        self.widget_plot_after.setGeometry(QtCore.QRect(10, 30, 601, 761))
        self.widget_plot_after.setObjectName("widget_plot_after")
        
        # 可视化标签
        self.label_head_plot = QtWidgets.QLabel(SubDenoise)
        self.label_head_plot.setGeometry(QtCore.QRect(620, 0, 1271, 51))
        font = QtGui.QFont()
        font.setFamily("等线 Light")
        font.setPointSize(16)
        self.label_head_plot.setFont(font)
        self.label_head_plot.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.label_head_plot.setAlignment(QtCore.Qt.AlignCenter)
        self.label_head_plot.setObjectName("label_head_plot")
        
        # 退出按钮
        self.pushButton_exit = QtWidgets.QPushButton(SubDenoise)
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

        self.retranslateUi(SubDenoise)
        self.pushButton_exit.clicked.connect(SubDenoise.close)
        QtCore.QMetaObject.connectSlotsByName(SubDenoise)

        # 连接信号和槽
        self.pushButton_data_in.clicked.connect(self.select_data_files)
        self.pushButton_plot_before.clicked.connect(self.plot_before)
        self.pushButton_plot_after.clicked.connect(self.plot_after)
        self.pushButton_begin.clicked.connect(self.start_denoise)
        self.comboBox_method.currentTextChanged.connect(self.show_method_info)

        # 初始化绘图
        self.setup_plotting()
        self.output_file_path = None  # 存储降噪后的文件路径

    def setup_plotting(self):
        """设置绘图区域"""
        # 降噪前绘图
        self.fig_before = plt.Figure()
        self.canvas_before = FC(self.fig_before)
        layout_before = QtWidgets.QVBoxLayout()
        layout_before.addWidget(self.canvas_before)
        toolbar_before = NavigationToolbar(self.canvas_before, self)
        layout_before.addWidget(toolbar_before)
        self.widget_plot_before.setLayout(layout_before)

        # 降噪后绘图
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
            "MSEED Files (*.mseed);;Seismic Data Files (*.mseed *.sac *.txt);;All Files (*.*)"
        )
        if files:
            self.listWidget_datafile_path.clear()
            self.listWidget_datafile_path.addItems(files)

    def plot_before(self):
        """绘制降噪前数据"""
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
            
            if file_path.endswith('.txt'):
                data = dfspy_cores.read_txt_array(file_path)
                dfspy_cores.plot_array(data, "Data Before Denoising", ax)
            else:
                st = dfspy_cores.read_stream(file_path)
                dfspy_cores.plot_stream(st, "Data Before Denoising", ax)
            
            self.canvas_before.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Plotting failed: {str(e)}")

    def plot_after(self):
        """绘制降噪后数据"""
        if not self.output_file_path:
            QMessageBox.warning(self, "Warning", "Please perform denoising first!")
            return

        try:
            # 导入核心模块绘图函数
            import sys
            sys.path.append('..')
            import dfspy_cores

            ax = self.fig_after.add_subplot(111)
            ax.clear()
            
            st = dfspy_cores.read_stream(self.output_file_path)
            dfspy_cores.plot_stream(st, "Data After Denoising", ax)
            
            self.canvas_after.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Plotting failed: {str(e)}")

    def show_method_info(self):
        """显示降噪方法信息"""
        method = self.comboBox_method.currentText()
        self.listWidget_params.clear()
        
        if method == "Bandpass":
            self.listWidget_params.addItem("Bandpass parameters:")
            self.listWidget_params.addItem("- Low cutoff frequency (Hz)")
            self.listWidget_params.addItem("- High cutoff frequency (Hz)")
            self.listWidget_params.addItem("- Filter order: 4")
        elif method == "Wavelet Denoise":
            self.listWidget_params.addItem("Wavelet denoise parameters:")
            self.listWidget_params.addItem("- Wavelet type: db4")
            self.listWidget_params.addItem("- Decomposition levels: 4")
        elif method == "Correlation Denoise":
            self.listWidget_params.addItem("Correlation denoise parameters:")
            self.listWidget_params.addItem("- Window length (window_size)")
            self.listWidget_params.addItem("- Step size (step_size)")
            self.listWidget_params.addItem("- Correlation threshold (corr_threshold)")
        elif method == "Spectral Subtraction":
            self.listWidget_params.addItem("Spectral subtraction parameters:")
            self.listWidget_params.addItem("- Noise frames")
            self.listWidget_params.addItem("- Alpha parameter")
            self.listWidget_params.addItem("- Beta parameter")

    def start_denoise(self):
        """开始降噪处理"""
        if self.listWidget_datafile_path.count() == 0:
            QMessageBox.warning(self, "Warning", "Please select data files first!")
            return

        method = self.comboBox_method.currentText()
        if not method:
            QMessageBox.warning(self, "Warning", "Please select a denoising method!")
            return

        file_path = self.listWidget_datafile_path.item(0).text()
        
        # 根据不同方法获取参数
        params = self.get_method_parameters(method)
        if params is None:
            return  # 用户取消了参数输入
        
        # 开始降噪处理
        method_key = {
            "带通滤波": "bandpass",
            "小波降噪": "wavelet", 
            "相关降噪": "correlation",
            "谱减法降噪": "spectral_subtraction"
        }.get(method)
        
        self.worker = DenoiseWorker(file_path, method_key, params)
        self.worker.finished.connect(self.on_denoise_finished)
        self.worker.error.connect(self.on_denoise_error)
        
        self.pushButton_begin.setEnabled(False)
        self.pushButton_begin.setText("Processing...")
        self.worker.start()

    def get_method_parameters(self, method):
        """获取降噪方法参数"""
        params = {}
        
        if method == "Bandpass":
            freqmin, ok1 = QInputDialog.getDouble(
                self, "Parameter Input", "Enter low cutoff frequency (Hz):", 20.0, 0.1, 1000.0, 1
            )
            if not ok1:
                return None
            
            freqmax, ok2 = QInputDialog.getDouble(
                self, "Parameter Input", "Enter high cutoff frequency (Hz):", 200.0, freqmin, 10000.0, 1
            )
            if not ok2:
                return None
                
            params = {'freqmin': freqmin, 'freqmax': freqmax}
            
        elif method == "Wavelet Denoise":
            wavelet, ok1 = QInputDialog.getText(
                self, "Parameter Input", "Enter wavelet type:", text="db4"
            )
            if not ok1:
                return None
                
            level, ok2 = QInputDialog.getInt(
                self, "Parameter Input", "Enter decomposition levels:", 4, 1, 10
            )
            if not ok2:
                return None
                
            params = {'wavelet': wavelet, 'level': level}
            
        elif method == "Correlation Denoise":
            window_size, ok1 = QInputDialog.getInt(
                self, "Parameter Input", "Enter window length:", 1024, 100, 50000
            )
            if not ok1:
                return None
                
            step_size, ok2 = QInputDialog.getInt(
                self, "Parameter Input", "Enter step size:", 512, 1, 1000
            )
            if not ok2:
                return None
                
            corr_threshold, ok3 = QInputDialog.getDouble(
                self, "Parameter Input", "Enter correlation threshold:", 0.5, 0.01, 1.0, 2
            )
            if not ok3:
                return None
                
            params = {
                'window_size': window_size,
                'step_size': step_size,
                'corr_threshold': corr_threshold
            }
            
        elif method == "Spectral Subtraction":
            noise_frames, ok1 = QInputDialog.getInt(
                self, "Parameter Input", "Enter number of noise frames:", 10, 1, 100
            )
            if not ok1:
                return None
                
            alpha, ok2 = QInputDialog.getDouble(
                self, "Parameter Input", "Enter Alpha parameter:", 1.0, 0.1, 10.0, 1
            )
            if not ok2:
                return None
                
            beta, ok3 = QInputDialog.getDouble(
                self, "Parameter Input", "Enter Beta parameter:", 0.02, 0.001, 1.0, 3
            )
            if not ok3:
                return None
                
            params = {
                'noise_frames': noise_frames,
                'alpha': alpha,
                'beta': beta,
                'window': 'hann',
                'frame_length': 1024,
                'hop_length': 512
            }
        
        return params

    def on_denoise_finished(self, output_path):
        """降噪完成回调"""
        self.pushButton_begin.setEnabled(True)
        self.pushButton_begin.setText("Start Denoising")
        self.output_file_path = output_path

        self.listWidget_params.addItem(f"Denoising finished! Output file: {output_path}")

        QMessageBox.information(self, "Success", "Denoising completed!")

    def on_denoise_error(self, error_message):
        """降噪错误回调"""
        self.pushButton_begin.setEnabled(True)
        self.pushButton_begin.setText("Start Denoising")
        QMessageBox.critical(self, "Error", error_message)

    def retranslateUi(self, SubDenoise):
        """设置 UI 文本"""
        _translate = QtCore.QCoreApplication.translate
        SubDenoise.setWindowTitle(_translate("SubDenoise", "Denoising"))
        self.label_head_oper.setText(_translate("SubDenoise", "Operations"))
        self.groupBox_open.setTitle(_translate("SubDenoise", "File Import"))
        self.pushButton_data_in.setText(_translate("SubDenoise", "Import Data"))
        self.pushButton_plot_before.setText(_translate("SubDenoise", "Plot Waveform"))
        self.groupBox_denoise.setTitle(_translate("SubDenoise", "Denoising"))
        self.label_select_method.setText(_translate("SubDenoise", "Select denoising method:"))
        self.pushButton_begin.setText(_translate("SubDenoise", "Start Denoising"))
        self.pushButton_plot_after.setText(_translate("SubDenoise", "Show After Denoising"))
        self.groupBox_visu_before.setTitle(_translate("SubDenoise", "Before Denoising"))
        self.groupBox_visu_after.setTitle(_translate("SubDenoise", "After Denoising"))
        self.label_head_plot.setText(_translate("SubDenoise", "Visualization"))
        self.pushButton_exit.setText(_translate("SubDenoise", "Exit"))


import imag_qrc_rc