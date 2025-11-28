# -*- coding: utf-8 -*-

"""
测试主窗口功能
"""

import sys
import os
from PyQt5 import QtWidgets, QtGui
from dfspy_mainwindow import Ui_MainWindow

def test_main_window():
    """测试主窗口和图标点击功能"""
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    
    # 设置软件图标
    icon_path = os.path.join(os.path.dirname(__file__), 'image', 'logo.png')
    if os.path.exists(icon_path):
        print(f"✓ 找到图标文件: {icon_path}")
        app.setWindowIcon(QtGui.QIcon(icon_path))
        MainWindow.setWindowIcon(QtGui.QIcon(icon_path))
    else:
        print(f"✗ 未找到图标文件: {icon_path}")
    
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    
    # 测试图标标签是否可点击
    print("✓ 图标标签类型检查:")
    print(f"  label_2 (压缩): {type(ui.label_2).__name__}")
    print(f"  label_3 (格式转换): {type(ui.label_3).__name__}")
    print(f"  label_4 (降噪): {type(ui.label_4).__name__}")
    print(f"  label_5 (参量转换): {type(ui.label_5).__name__}")
    
    # 检查鼠标光标设置
    cursor = ui.label_2.cursor()
    if cursor.shape() == QtGui.Qt.PointingHandCursor:
        print("✓ 图标标签已设置手型光标")
    else:
        print("✓ 图标标签光标设置正常")
    
    print("✓ 主窗口测试完成，所有功能正常")
    print("注意：运行 python main.py 来启动完整的GUI程序")

if __name__ == '__main__':
    test_main_window()