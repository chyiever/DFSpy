# -*- coding: utf-8 -*-

'''
**************************************************************************************
Software Name : Distributed Fiber-Optic Sensing Data Processing Software [DFSPy] V1.0
Copyright     : Institute of Semiconductors, Chinese Academy of Sciences (Software Copyright No: 2025SR0353448)
Contact Email : qi.gh@outlook.com (Developer: https://github.com/chyiever)
Language      : Python 3.9+
Description   : DFSPy is a professional data-processing tool for distributed fiber-optic sensing, designed for researchers to provide efficient and reliable processing solutions. The software integrates advanced algorithms to support preprocessing, analysis, and visualization of data collected by distributed fiber-optic sensing systems.
Usage Notice  :
1. This software is an open-source research tool developed by the Institute of Semiconductors, Chinese Academy of Sciences. It is intended for academic research and non-commercial use only. Users must comply with applicable laws and research ethics and not use it for commercial or illegal activities.
2. The software is provided "as is". The developers make no express or implied warranties regarding suitability, completeness, or accuracy.
3. If you cite this software in academic publications, please acknowledge the software source and the developers.
***************************************************************************************
'''

"""
DFSPy GUI entry point

Launch the DFSPy graphical user interface for distributed fiber-optic sensing data processing.
"""

import sys
import os,matplotlib, obspy
from PyQt5 import QtWidgets, QtGui
from dfspy_mainwindow import Ui_MainWindow
import requests #新增
import matplotlib #新增
matplotlib.use('QtAgg') #新增,需要安装PyQt5

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    
    # 设置软件图标
    icon_path = os.path.join(os.path.dirname(__file__), 'image', 'logo.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QtGui.QIcon(icon_path))
        MainWindow.setWindowIcon(QtGui.QIcon(icon_path))
    
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
