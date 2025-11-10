import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from PIL import Image, ImageTk
import matplotlib.ticker as ticker  # 导入 ticker 模块
import threading
import serial
import time
from matplotlib.font_manager import FontProperties
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 设置支持中文的字体
font = FontProperties(fname="C:/Windows/Fonts/simhei.ttf")  # 确保路径正确

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义卷积神经网络模型类
class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 12, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv_output_size = input_size // 2  # 每次池化减半后的大小
        self.fc1 = nn.Linear(12 * self.conv_output_size, 24)
        self.fc2 = nn.Linear(24, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = x.view(-1, 12 * self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_path, input_size, num_classes):
    """加载训练好的模型"""
    model = CNN(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_input(input_data, scaler_path):
    """预处理输入数据"""
    sc = joblib.load(scaler_path)
    input_data = np.array(input_data).reshape(1, 2, -1)
    input_data = sc.transform(input_data.reshape(-1, input_data.shape[-1])).reshape(input_data.shape)
    input_tensor = torch.from_numpy(input_data.astype(np.float32)).to(device)
    return input_tensor

def predict(model, input_tensor):
    """使用模型进行预测"""
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, dim=1)
    return predicted.item()

class SerialDataCollectorAndPredictor:
    def __init__(self, master):
        self.master = master
        self.collected_data = pd.DataFrame()  # 用于保存每次单个输入到模型的数据
        self.collected_data1 = pd.DataFrame()  # 用于保存连续模式下的数据
        self.collected_data2 = pd.DataFrame()  # 用于保存非连续模式下的数据
        self.continuous_mode = tk.BooleanVar(value=True)  # 控制是否是连续模式，默认连续模式

        # 其他初始化代码...


        master.title("Real Time Typing System")
        master.configure(bg='#f0f0f0')
        master.bind('<space>', self.on_start_button_click)

        # 左侧框架
        self.left_frame = tk.Frame(master)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        # 串口号
        self.label = tk.Label(self.left_frame, text="Serial Number:", font=("Arial", 12, "bold"), fg="#333333", bg="#f0f0f0")
        self.label.grid(row=0, column=0, sticky="e", padx=(10, 10), pady=(5, 5))

        self.port_entry = tk.Entry(self.left_frame, width=10, font=("Arial", 12), fg="#333333", bg="#ffffff", relief="sunken", bd=2)
        self.port_entry.insert(0, "COM9")
        self.port_entry.grid(row=0, column=1, padx=(10, 10), pady=(5, 5), sticky="w")

        # 波特率
        self.baud_label = tk.Label(self.left_frame, text="Baud-rate:", font=("Arial", 12, "bold"), fg="#333333", bg="#f0f0f0")
        self.baud_label.grid(row=1, column=0, sticky="e", padx=(10, 10), pady=(5, 5))

        self.baud_entry = tk.Entry(self.left_frame, width=10, font=("Arial", 12), fg="#333333", bg="#ffffff", relief="sunken", bd=2)
        self.baud_entry.insert(0, "115200")
        self.baud_entry.grid(row=1, column=1, padx=(10, 10), pady=(5, 5), sticky="w")

        # 采集时间
        self.duration_label = tk.Label(self.left_frame, text="Acquisition Time (s):", font=("Arial", 12, "bold"), fg="#333333", bg="#f0f0f0")
        self.duration_label.grid(row=2, column=0, sticky="e", padx=(10, 10), pady=(5, 5))

        self.duration_entry = tk.Entry(self.left_frame, width=10, font=("Arial", 12), fg="#333333", bg="#ffffff", relief="sunken", bd=2)
        self.duration_entry.insert(0, "2.5")
        self.duration_entry.grid(row=2, column=1, padx=(10, 10), pady=(5, 5), sticky="w")

        # 波形图显示区
        self.figure, self.ax = plt.subplots(figsize=(4, 4))
        self.ax.set_facecolor("#f5f5f5")  # 设置更浅的图表背景颜色

        self.canvas = FigureCanvasTkAgg(self.figure, self.left_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=4, column=0, columnspan=3, pady=(10, 10), padx=(10, 10), sticky="n")

        # 在 Canvas 上创建一个 Frame 用来容纳 Mode 选择控件
        self.mode_frame_on_canvas = tk.Frame(self.canvas_widget, bg="#e0e0e0")

        # 将 Frame 放置在 Canvas 上 (x, y 为 Frame 的位置)
        self.canvas.get_tk_widget().create_window(10, 10, window=self.mode_frame_on_canvas, anchor="nw")

        # Mode 选择控件放置在这个 Frame 中
        self.mode_label = tk.Label(self.mode_frame_on_canvas, text="Mode:", font=("Arial", 6, "bold"), fg="#333333",
                                   bg="#e0e0e0")
        self.mode_label.pack(side=tk.LEFT, padx=5)

        self.radio_continuous = tk.Radiobutton(self.mode_frame_on_canvas, text="Continuous",
                                               variable=self.continuous_mode, value=True,
                                               font=("Arial", 6), fg="#333333", bg="#e0e0e0")
        self.radio_continuous.pack(side=tk.LEFT, padx=5)

        self.radio_non_continuous = tk.Radiobutton(self.mode_frame_on_canvas, text="Non-Continuous",
                                                   variable=self.continuous_mode, value=False,
                                                   font=("Arial", 6), fg="#333333", bg="#e0e0e0")
        self.radio_non_continuous.pack(side=tk.LEFT, padx=5)

        # 右侧框架
        self.right_frame = tk.Frame(master)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # 创建一个新的框架来包含模型选择和start按钮
        self.control_frame = tk.Frame(self.right_frame, bg="#f0f0f0", bd=2, relief="groove")
        self.control_frame.grid(row=0, column=0, sticky="n", padx=5, pady=10)

        # 模型选择控件
        self.model_label = tk.Label(self.control_frame, text="Select Model:", font=("Arial", 12, "bold"), fg="#333333", bg="#f0f0f0")
        self.model_label.grid(row=0, column=0, sticky="e", padx=5, pady=11)

        model_options = ["CNN-TCN", "CNN", "TCN"]
        self.selected_model = tk.StringVar()
        self.selected_model.set(model_options[0])

        self.model_menu = tk.OptionMenu(self.control_frame, self.selected_model, *model_options)
        self.model_menu.config(font=("Arial", 12), fg="#333333", bg="#ffffff", bd=2, relief="sunken")
        self.model_menu.grid(row=0, column=1, padx=(5, 10), pady=5, sticky="w")

        # 处理不同 Pillow 版本的 LANCZOS 插值方法
        try:
            from PIL import Resampling
            resampling_method = Resampling.LANCZOS
        except ImportError:
            resampling_method = Image.LANCZOS

        # 加载并缩放 start_icon
        start_icon = Image.open("START.png")
        start_icon = start_icon.resize((start_icon.width // 6, start_icon.height // 6), resampling_method)
        start_icon = ImageTk.PhotoImage(start_icon)

        self.start_button = tk.Button(
            self.control_frame,
            image=start_icon,
            command=self.on_start_button_click,
            borderwidth=0
        )
        self.start_button.grid(row=0, column=2, padx=(5, 0), pady=(0, 0), sticky="w")

        self.start_button.image = start_icon  # 保持对图像对象的引用

        # 加载并缩放 translation_icon
        translation_icon = Image.open("conversion.png")
        translation_icon = translation_icon.resize((translation_icon.width // 12, translation_icon.height // 12),
                                                   resampling_method)
        translation_icon = ImageTk.PhotoImage(translation_icon)

        self.result_label = tk.Label(
            self.right_frame,
            image=translation_icon,
            borderwidth=0,
        )
        self.result_label.grid(row=3, column=0, pady=(6), sticky="ew")

        self.result_label.image = translation_icon  # 保持对图像对象的引用

        # 创建文本框，不改变大小
        self.result_text = tk.Text(self.right_frame, height=6, width=40)
        self.result_text.grid(row=4, column=0, pady=5)

        self.result_text.tag_configure("custom", font=("Helvetica", 30), foreground="red")

        self.frame = tk.Frame(self.right_frame, borderwidth=2, relief=tk.SUNKEN)
        self.frame.grid(row=1, column=0, padx=5, pady=23)

        self.canvas_img = tk.Canvas(
            self.frame,
            width=600,
            height=550,
            borderwidth=2,
            relief="ridge",
            highlightthickness=2,
            highlightbackground="black"
        )

        self.canvas_img.pack()

        # 设置路径
        self.model_path = 'models/best_cnn_model.pth1'
        self.scaler_path = 'scaler.pkl'

        # 定义num_classes和input_size
        self.num_classes = 4  # 根据训练时的类别数量修改
        self.input_size = 250  # 根据训练时的输入大小修改

        # 加载模型和标准化器
        self.model = load_model(self.model_path, self.input_size, self.num_classes)
        # 处理不同 Pillow 版本的 LANCZOS 插值方法
        try:
            from PIL import Resampling
            resampling_method = Resampling.LANCZOS
        except ImportError:
            resampling_method = Image.LANCZOS

        # 在右侧添加新的框架用于显示图片
        self.image_frame = tk.Frame(master)
        self.image_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")

        # 加载并显示图片
        image_path = "img.png"  # 将 "your_image.png" 替换为您的图片路径

        try:
            # 加载并调整图片大小
            self.right_side_image = Image.open(image_path)
            self.right_side_image = self.right_side_image.resize((1000, 1080), resampling_method)

            # 转换为 Tkinter 可用的 PhotoImage
            self.right_side_photo = ImageTk.PhotoImage(self.right_side_image)

            # 创建标签来显示图片
            self.image_label = tk.Label(self.image_frame, image=self.right_side_photo)

            # 放置标签
            self.image_label.pack()
        except Exception as e:
            print(f"加载图片时出错: {e}")

    def on_start_button_click(self, event=None):  # 修改签名，event 参数是为了处理 Tkinter 的事件
        threading.Thread(target=self.collect_and_predict).start()  # 使用多线程防止UI卡顿

    def start_collection(self):
        port = self.port_entry.get()
        baud_rate = int(self.baud_entry.get())
        collection_duration = float(self.duration_entry.get())

        try:
            ser = serial.Serial(port, baud_rate, timeout=1)
            time.sleep(2)  # 等待串口初始化
        except serial.SerialException:
            messagebox.showerror("错误", "无法打开串口。请检查串口号是否正确。")
            return

        start_time = time.time()  # 记录开始时间
        data = []  # 存储数据的列表

        try:
            while True:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    values = line.split(',')  # 分割数据
                    data.append([float(v) for v in values])  # 将数据添加到列表中，并转换为浮点数
                    print(f"Received: {line}")

                # 检查是否超过数据采集时间
                if time.time() - start_time > collection_duration:
                    print("Data collection finished.")
                    break
        except KeyboardInterrupt:
            print("Program interrupted by user.")
        finally:
            # 关闭串口
            ser.close()

        def smooth_series(series, window_size=10):
            return series.rolling(window=window_size, min_periods=1).mean()

        if data:
            df = pd.DataFrame(data).iloc[:, :-3]  # 创建 DataFrame 并删除后三列
            df = df.T  # 转置数据
            df = df.iloc[:, 1:251]  # 选择从第二列到第251列（保留这些列）
            df = df.apply(lambda x: x - x.iloc[0], axis=1)  # 每一行减去当前的第一列的值
            df = df.apply(smooth_series, axis=1)  # 对每行进行平滑处理
            # 保存到 CSV 文件
            df.to_csv('collected_data.csv', header=False, index=False)
            if self.continuous_mode.get():
                # 如果是连续模式，将新数据追加到历史数据中
                self.collected_data1 = pd.concat([self.collected_data1, df], axis=1)
                # 绘制连续模式的波形图
                self.plot_waveform(self.collected_data1)
            else:
                # 如果是非连续模式，只显示当前的采集数据
                self.collected_data2 = df
                # 绘制非连续模式的波形图
                self.plot_waveform(self.collected_data2)
        else:
            print("No data collected.")

    def plot_waveform(self, df):
        self.ax.clear()

        labels = ['Index ·', 'Middle -']  # 自定义标签

        for i in range(len(df)):
            self.ax.plot(range(df.shape[1]), df.iloc[i], label=labels[i])

        # 设置纵坐标格式化，只显示小数点后一位
        self.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        # 使用自定义字体属性美化标题、X轴和Y轴标签
        title_font = {'family': 'Arial', 'color': 'darkblue', 'weight': 'bold', 'size': 16}
        label_font = {'family': 'Arial', 'color': 'black', 'weight': 'bold', 'size': 14}

        # 创建 FontProperties 对象用于图例字体美化
        legend_font = FontProperties(family='Arial', weight='bold', size=6)

        self.ax.set_title('Waveform figure', fontdict=title_font)
        self.ax.set_xlabel('Sample point', fontdict=label_font)
        self.ax.set_ylabel(r'$\Delta C$', fontdict=label_font)  # 使用 LaTeX 表达式

        self.figure.tight_layout(pad=2.0)  # 调整布局，避免文本被遮挡

        # 自动选择最佳位置放置图例
        self.ax.legend(prop=legend_font, loc='best')

        self.canvas.draw()

    def collect_and_predict(self):
        self.start_collection()
        self.predict_and_display()

    def predict_and_display(self):
        # 读取采集的数据
        df = pd.read_csv('collected_data.csv', header=None)
        input_data = df.values.tolist()  # 将DataFrame转换为列表

        # 预处理输入数据
        input_tensor = preprocess_input(input_data, self.scaler_path)

        # 进行预测
        prediction = predict(self.model, input_tensor)
        print(f'Predicted class: {prediction}')

        # 定义预测结果对应的字符
        result_characters = ["E", "H", "L", "O", "E", "F", "H", "I", "O"]
        #result_characters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        # 在文本框中追加显示预测结果
        if prediction < len(result_characters):
            self.result_text.insert(tk.END, result_characters[prediction], "custom")  # 追加预测字符
        else:
            self.result_text.insert(tk.END, "未知", "custom")  # 如果预测结果超出范围

        # 定义预测结果对应的图像文件路径
        image_files = ["E.png", "H.png", "L.png", "O.png", "E.png", "F.png", "H.png", "I.png", "O.png"]  # 按照数据集行索引的顺序排列
        #image_files = ["A.png", "B.png", "C.png", "D.png", "E.png", "F.png", "G.png", "H.png", "I.png", "J.png", "K.png", "L.png", "M.png", "N.png", "O.png", "P.png", "Q.png", "R.png", "S.png", "T.png", "U.png", "V.png", "W.png", "X.png", "Y.png", "Z.png"]  # 按照数据集行索引的顺序排列
        # 根据预测结果选择对应的图像文件
        if prediction < len(image_files):
            image_path = image_files[prediction]
            # 加载图像
            image = Image.open(image_path)
            # 获取画布的大小
            canvas_width = self.canvas_img.winfo_width()
            canvas_height = self.canvas_img.winfo_height()
            # 调整图像大小，使其铺满画布，不保持纵横比
            image = image.resize((canvas_width, canvas_height), Image.LANCZOS)
            # 将图像转换为Tkinter可用格式
            tk_image = ImageTk.PhotoImage(image)

            # 清除画布中的任何已有图像
            self.canvas_img.delete("all")
            # 更新画布中的图像
            self.canvas_img.create_image(0, 0, anchor=tk.NW, image=tk_image)
            self.canvas_img.image = tk_image  # 保持对图像对象的引用
        else:
            print("预测结果超出图像范围")


# 创建Tkinter窗口
root = tk.Tk()

# 设置缩放比例，例如设置为2.0表示放大一倍
root.tk.call('tk', 'scaling', 3.0)

app = SerialDataCollectorAndPredictor(root)
root.mainloop()