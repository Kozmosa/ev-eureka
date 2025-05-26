import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

# 设置中文显示
plt.rcParams["font.family"] = ["MiSans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def plot_csv_columns(file1, file2, output_dir=None, title=None, show=True, sort=True):
    """
    从两个CSV文件中读取数据并绘制指定列的折线图
    
    参数:
        file1 (str): 第一个CSV文件路径
        file2 (str): 第二个CSV文件路径
        output_dir (str, optional): 图表保存目录
        title (str, optional): 图表标题
        show (bool, optional): 是否显示图表
        sort (bool, optional): 是否在绘图前对数据进行排序
    """
    # 读取数据
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return
    
    # 检查所需列是否存在
    required_columns = ['PREMIUM', 'predicted_premium']
    for col in required_columns:
        if col not in df1.columns:
            print(f"错误: 文件 {file1} 中缺少列 '{col}'")
            return
        if col not in df2.columns:
            print(f"错误: 文件 {file2} 中缺少列 '{col}'")
            return
        
    # 如果指定了排序，则对数据进行排序
    if sort:
        df1 = df1.sort_values(by='PREMIUM')
        df2 = df2.sort_values(by='PREMIUM')
        sort_suffix = "（已排序）"
    else:
        sort_suffix = ""
    
    # 创建图表
    plt.figure(figsize=(14, 8))
    
    # 绘制第一个文件的数据
    plt.subplot(2, 1, 1)
    plt.plot(df1['PREMIUM'], label='实际保费', color='blue', alpha=0.7)
    plt.plot(df1['predicted_premium'], label='预测保费', color='red', alpha=0.7)
    plt.title(f"{os.path.basename(file1)} 的保费数据对比")
    plt.xlabel('样本索引')
    plt.ylabel('保费金额')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制第二个文件的数据
    plt.subplot(2, 1, 2)
    plt.plot(df2['PREMIUM'], label='实际保费', color='blue', alpha=0.7)
    plt.plot(df2['predicted_premium'], label='预测保费', color='red', alpha=0.7)
    plt.title(f"{os.path.basename(file2)} 的保费数据对比")
    plt.xlabel('样本索引')
    plt.ylabel('保费金额')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加总体标题
    if title:
        plt.suptitle(title, fontsize=16)
    else:
        plt.suptitle('保费数据与预测保费对比分析', fontsize=16)
    
    # 自动调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
    
    # 保存图表
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"保费对比_{timestamp}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    # 显示图表
    if show:
        plt.show()
    
    return plt


def plot_single_csv_columns(file1, output_dir=None, title=None, show=True, sort=True):
    """
    从两个CSV文件中读取数据并绘制指定列的折线图
    
    参数:
        file1 (str): 第一个CSV文件路径
        output_dir (str, optional): 图表保存目录
        title (str, optional): 图表标题
        show (bool, optional): 是否显示图表
        sort (bool, optional): 是否在绘图前对数据进行排序
    """
    # 读取数据
    try:
        df1 = pd.read_csv(file1)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return
    
    # 检查所需列是否存在
    required_columns = ['PREMIUM', 'predicted_premium']
    for col in required_columns:
        if col not in df1.columns:
            print(f"错误: 文件 {file1} 中缺少列 '{col}'")
            return
        
    # 如果指定了排序，则对数据进行排序
    if sort:
        df1 = df1.sort_values(by='PREMIUM')
        sort_suffix = "（已排序）"
    else:
        sort_suffix = ""
    
    # 创建图表
    plt.figure(figsize=(14, 8))
    
    # 绘制第一个文件的数据
    plt.subplot(2, 1, 1)
    plt.plot(df1['PREMIUM'], label='实际保费', color='blue', alpha=0.7)
    plt.plot(df1['predicted_premium'], label='预测保费', color='red', alpha=0.7)
    plt.title(f"{os.path.basename(file1)} 的保费数据对比")
    plt.xlabel('样本索引')
    plt.ylabel('保费金额')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加总体标题
    if title:
        plt.suptitle(title, fontsize=16)
    else:
        plt.suptitle('保费数据与预测保费对比分析', fontsize=16)
    
    # 自动调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
    
    # 保存图表
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"保费对比_{timestamp}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    # 显示图表
    if show:
        plt.show()
    
    return plt

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='绘制两个CSV文件中premium和predicted_premium列的折线图')
    parser.add_argument('file1', help='第一个CSV文件路径')
    # parser.add_argument('file2', help='第二个CSV文件路径', default=None)
    parser.add_argument('-o', '--output', help='图表保存目录')
    parser.add_argument('-t', '--title', help='图表标题')
    parser.add_argument('--no-show', action='store_true', help='不显示图表')
    parser.add_argument('--sort', action='store_true', help='在绘图前对数据进行排序')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查文件是否存在
    # for file_path in [args.file1, args.file2]:
    #     if not os.path.exists(file_path):
    #         print(f"错误: 文件 '{file_path}' 不存在")
    #         return
    
    # 绘制图表
    # if args.file2 is not None:
    #     plot_csv_columns(
    #         args.file1, 
    #         args.file2, 
    #         output_dir=args.output, 
    #         title=args.title, 
    #         show=not args.no_show,
    #         sort=args.sort
    #     )
    # else:
    plot_single_csv_columns(
        args.file1, 
        output_dir=args.output, 
        title=args.title, 
        show=not args.no_show,
        sort=args.sort
    )

if __name__ == "__main__":
    main()    