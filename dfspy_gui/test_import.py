# -*- coding: utf-8 -*-

"""
测试 GUI 模块导入
"""

print("测试GUI模块导入...")

try:
    print("导入主窗口模块...")
    import dfspy_mainwindow
    print("✓ 主窗口模块导入成功")
    
    print("导入格式转换子窗口...")
    import dfspy_subwindow_formatc
    print("✓ 格式转换子窗口导入成功")
    
    print("导入压缩子窗口...")
    import dfspy_subwindow_compress
    print("✓ 压缩子窗口导入成功")
    
    print("导入降噪子窗口...")
    import dfspy_subwindow_denoise
    print("✓ 降噪子窗口导入成功")
    
    print("导入参量转换子窗口...")
    import dfspy_subwindow_parac
    print("✓ 参量转换子窗口导入成功")
    
    print("\n所有GUI模块导入成功！")
    
except Exception as e:
    print(f"导入失败：{e}")
    import traceback
    traceback.print_exc()