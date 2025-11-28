# -*- coding: utf-8 -*-

"""
测试绘图标题是否都已改为英文
"""

import re

def check_chinese_plot_titles():
    """检查绘图标题是否还有中文"""
    
    files_to_check = [
        'dfspy_subwindow_compress.py',
        'dfspy_subwindow_denoise.py', 
        'dfspy_subwindow_formatc.py',
        'dfspy_subwindow_parac.py'
    ]
    
    chinese_found = False
    
    for filename in files_to_check:
        print(f"\n检查文件: {filename}")
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 查找 plot_array 和 plot_stream 调用中的中文
            plot_calls = re.findall(r'(plot_array|plot_stream)\([^)]+["\']([^"\']*[\u4e00-\u9fff][^"\']*)["\'][^)]*\)', content)
            
            if plot_calls:
                print(f"  发现中文绘图标题:")
                for call_type, title in plot_calls:
                    print(f"    {call_type}: '{title}'")
                chinese_found = True
            else:
                print(f"  ✓ 未发现中文绘图标题")
                
        except FileNotFoundError:
            print(f"  文件未找到: {filename}")
        except Exception as e:
            print(f"  检查文件时出错: {e}")
    
    if not chinese_found:
        print("\n✅ 所有绘图标题已改为英文！")
    else:
        print("\n❌ 仍有中文绘图标题需要修改")

if __name__ == "__main__":
    check_chinese_plot_titles()