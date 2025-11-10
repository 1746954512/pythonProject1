import os
import serial
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QGraphicsView, QGraphicsScene, \
    QDockWidget, QLabel
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
import pyqtgraph as pg
from pyts.image import GramianAngularField
from PyQt5.QtGui import QImage, QPixmap
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from PyQt5.QtGui import QFont
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn.utils as nn_utils
import timm
import asyncio
import websockets
import json


# 定义一个信号类来传递 predicted_class
class WebSocketThread(QThread):
    predicted_class_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.predicted_class = None
        self.pause_sending = False  # 添加一个状态变量，控制是否暂停发送

        self.predicted_class_signal.connect(self.update_predicted_class)

    def update_predicted_class(self, mapped_output):
        """接收来自主线程的 predicted_class 信号"""
        self.predicted_class = mapped_output
        print(f"接收到主线程的 predicted_class: {mapped_output}")

    async def recognize(self, predicted_class):
        """识别函数，返回传入的 predicted_class"""
        return predicted_class

    async def handler(self, websocket):
        """WebSocket 处理函数"""
        print("Received connection")
        while True:
            result = await self.recognize(self.predicted_class if self.predicted_class is not None else "")
            await websocket.send(json.dumps({"result": result}))
            print(f"发送消息: {result}")
            await asyncio.sleep(0.1)  # 每 0.1 秒检查一次

    async def main(self):
        """主事件循环"""
        server = await websockets.serve(self.handler, "192.168.77.252", 8765)
        print("WebSocket 服务器已启动，地址：ws://192.168.77.252:8765")
        await asyncio.Future()

    def run(self):
        """在 WebSocket 线程中运行事件循环"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.main())

    def toggle_pause(self):
        """切换暂停状态"""
        self.pause_sending = not self.pause_sending
        print(f"数据发送状态: {'暂停' if self.pause_sending else '继续'}")


# 滑动窗口平滑函数
def smooth_data(data, window_size=10):
    # 创建一个简单的均值滤波器
    kernel = np.ones(window_size) / window_size
    return convolve(data, kernel, mode='nearest')


class SerialThread(QThread):
    data_received = pyqtSignal(list)

    def __init__(self, port, baudrate=9600):
        super().__init__()
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=1)

    def run(self):
        while True:
            if self.ser.in_waiting > 0:
                data = self.ser.readline()
                try:
                    waveform_values = list(map(float, data.decode('utf-8').strip().split(',')))
                    self.data_received.emit(waveform_values)
                except ValueError:
                    pass


# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义 EfficientNet 模块
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        features = self.efficientnet.features(x)
        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        return pooled_features


# 定义 MobileNetV2 模块
class MobileNetV2Model(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2Model, self).__init__()
        self.mobilenetv2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.mobilenetv2.classifier[1] = nn.Linear(self.mobilenetv2.classifier[1].in_features, num_classes)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        features = self.mobilenetv2.features(x)
        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        return pooled_features


# 定义整合模型
class CombinedModel(nn.Module):
    def __init__(self, num_classes):
        super(CombinedModel, self).__init__()
        self.efficientnet_model = EfficientNetModel(num_classes=num_classes)  # 处理 GASF
        self.mobilenetv2_model = MobileNetV2Model(num_classes=num_classes)  # 处理 GADF
        self.fc = nn.Linear(1280 + 1280, num_classes)  # 拼接后的特征维度

    def forward(self, gasf, gadf):
        # 将 GASF 数据送入 EfficientNet
        efficientnet_out = self.efficientnet_model(gasf)
        # 将 GADF 数据送入 MobileNetV2
        mobilenetv2_out = self.mobilenetv2_model(gadf)
        # 拼接 EfficientNet 和 MobileNetV2 的输出
        combined = torch.cat((efficientnet_out, mobilenetv2_out), dim=1)
        # 通过全连接层进行分类
        out = self.fc(combined)
        return out


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("5 Waveform Plot Display")
        self.setGeometry(100, 100, 1500, 600)
        self.original_size = self.size()  # 保存原始窗口尺寸
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # 左侧GADF显示区
        left_gadf_layout = QVBoxLayout()
        self.gadf_canvas_widgets = [QGraphicsView() for _ in range(5)]
        for canvas_widget in self.gadf_canvas_widgets:
            canvas_widget.setScene(QGraphicsScene())
            canvas_widget.setFixedSize(100, 100)
            left_gadf_layout.addWidget(canvas_widget)

        left_gadf_widget = QWidget()
        left_gadf_widget.setLayout(left_gadf_layout)
        main_layout.addWidget(left_gadf_widget)

        # 中间波形显示区
        center_waveform_layout = QVBoxLayout()
        waveform_names = ["Thumb", "Index", "Middle", "Ring", "Little"]

        self.plot_widgets = [pg.PlotWidget() for _ in range(5)]
        for i, plot_widget in enumerate(self.plot_widgets):
            font = QFont('Arial', 12, QFont.Bold)  # 设置字体、大小和加粗
            plot_widget.setTitle(waveform_names[i], color='white', size='14pt', font=font)
            plot_widget.setTitle(waveform_names[i])  # 设置标题
            plot_widget.addLegend()  # 添加图例
            plot_widget.plot([], [], name=waveform_names[i])  # 设置曲线名称
            center_waveform_layout.addWidget(plot_widget)
        center_waveform_widget = QWidget()
        center_waveform_widget.setLayout(center_waveform_layout)
        main_layout.addWidget(center_waveform_widget)

        # 右侧GASF显示区
        right_gasf_layout = QVBoxLayout()
        self.gasf_canvas_widgets = [QGraphicsView() for _ in range(5)]
        for canvas_widget in self.gasf_canvas_widgets:
            canvas_widget.setScene(QGraphicsScene())
            canvas_widget.setFixedSize(100, 100)
            right_gasf_layout.addWidget(canvas_widget)

        right_gasf_widget = QWidget()
        right_gasf_widget.setLayout(right_gasf_layout)
        main_layout.addWidget(right_gasf_widget)

        # 左侧大画布
        self.left_large_canvas = QGraphicsView()
        self.left_large_canvas.setScene(QGraphicsScene())
        self.left_large_canvas.setFixedSize(400, 400)

        left_dock_widget = QDockWidget("Fused GADF Image", self)
        left_dock_container = QWidget()
        left_dock_layout = QVBoxLayout(left_dock_container)

        # 添加第一个画布（QGraphicsView）
        left_dock_layout.addWidget(self.left_large_canvas)

        # 创建一个水平布局用于 gesture 和 translate 文本框
        text_layout_left = QHBoxLayout()

        # 添加 gesture 和 translate 文本框注释
        self.left_gesture_label = QLabel("Gesture: ")
        self.left_gesture_label.setFont(QFont('Arial', 18, QFont.Bold))  # 设置字体
        text_layout_left.addWidget(self.left_gesture_label, alignment=Qt.AlignCenter)  # 居中对齐

        # 将水平布局添加到垂直布局中
        left_dock_layout.addLayout(text_layout_left)

        # 添加第二个画布（QLabel 显示图片）
        self.left_image_label = QLabel()
        self.left_image_label.setPixmap(QPixmap("1.png").scaled(400, 200, Qt.KeepAspectRatio))
        left_dock_layout.addWidget(self.left_image_label, alignment=Qt.AlignCenter)  # 居中对齐

        left_dock_widget.setWidget(left_dock_container)
        left_dock_widget.setFloating(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock_widget)

        # 右侧大画布
        self.right_large_canvas = QGraphicsView()
        self.right_large_canvas.setScene(QGraphicsScene())
        self.right_large_canvas.setFixedSize(400, 400)

        right_dock_widget = QDockWidget("Fused GASF Image", self)
        right_dock_container = QWidget()
        right_dock_layout = QVBoxLayout(right_dock_container)

        # 添加第一个画布（QGraphicsView）
        right_dock_layout.addWidget(self.right_large_canvas)

        # 创建一个水平布局用于 gesture 和 translate 文本框
        text_layout_right = QHBoxLayout()

        # 添加 gesture 和 translate 文本框注释

        self.right_translate_label = QLabel("Translate: ")
        self.right_translate_label.setFont(QFont('Arial', 18, QFont.Bold))  # 设置字体
        text_layout_right.addWidget(self.right_translate_label, alignment=Qt.AlignCenter)  # 居中对齐

        # 将水平布局添加到垂直布局中
        right_dock_layout.addLayout(text_layout_right)

        # 添加第二个画布（QLabel 显示图片）
        self.right_image_label = QLabel()
        self.right_image_label.setPixmap(QPixmap("2.png").scaled(400, 200, Qt.KeepAspectRatio))
        right_dock_layout.addWidget(self.right_image_label, alignment=Qt.AlignCenter)  # 居中对齐

        right_dock_widget.setWidget(right_dock_container)
        right_dock_widget.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock_widget)

        # 定时器用于更新显示
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(100)

        # 定时器用于检查数据长度并触发推理
        self.check_data_timer = QTimer(self)
        self.check_data_timer.timeout.connect(self.check_data_length)
        self.check_data_timer.start(2000)  # 每 100 毫秒检查一次

        self.waveforms_data = [[] for _ in range(5)]
        self.gasf_images = [np.zeros((224, 224)) for _ in range(5)]
        self.gadf_images = [np.zeros((224, 224)) for _ in range(5)]
        self.gasf = GramianAngularField(image_size=224, method='summation')
        self.gadf = GramianAngularField(image_size=224, method='difference')

        self.serial_thread = SerialThread('COM9')
        self.serial_thread.data_received.connect(self.update_waveform_data)
        self.serial_thread.start()
        # 初始化神经网络
        self.num_classes = 5  # 假设有5个类别
        self.model = CombinedModel(num_classes=self.num_classes).to(device)
        # 加载预训练权重
        self.model.load_state_dict(torch.load("best_combined_model2.pth"))
        self.model.eval()  # 设置为评估模式
        # 设置窗口可以接收键盘事件
        self.setFocusPolicy(Qt.StrongFocus)

        # 确保在主线程初始化时创建 WebSocket 线程
        self.websocket_thread = WebSocketThread()
        self.websocket_thread.start()

        # 缓冲区管理
        self.buffer_size = 224  # 缓冲区大小
        self.waveforms_buffer = [[] for _ in range(5)]  # 5 个通道的缓冲区
        self.is_inferring = False  # 标记是否正在推理，避免重复推理

    def resizeEvent(self, event):
        """窗口大小变化事件处理"""
        super().resizeEvent(event)
        self.scale_widgets()

    def scale_widgets(self):
        """等比列放大所有控件"""
        new_size = self.size()
        width_scale = new_size.width() / self.original_size.width()
        height_scale = new_size.height() / self.original_size.height()

        # 缩放所有控件
        for canvas_widget in self.gadf_canvas_widgets:
            canvas_widget.setFixedSize(100 * width_scale, 100 * height_scale)

        for canvas_widget in self.gasf_canvas_widgets:
            canvas_widget.setFixedSize(100 * width_scale, 100 * height_scale)

        self.left_large_canvas.setFixedSize(400 * width_scale, 400 * height_scale)
        self.right_large_canvas.setFixedSize(400 * width_scale, 400 * height_scale)

        self.left_image_label.setPixmap(
            QPixmap("9.png").scaled(400 * width_scale, 200 * height_scale, Qt.KeepAspectRatio))
        self.right_image_label.setPixmap(
            QPixmap("10.png").scaled(400 * width_scale, 200 * height_scale, Qt.KeepAspectRatio))

    def update_waveform_data(self, waveform_values):
        """更新波形数据并填充缓冲区"""
        if self.is_inferring:  # 如果正在推理，不更新缓冲区
            return

        for i in range(min(5, len(waveform_values))):
            self.waveforms_buffer[i].append(waveform_values[i])
            if len(self.waveforms_buffer[i]) > self.buffer_size:
                self.waveforms_buffer[i].pop(0)

    def update_display(self):
        """更新波形显示"""
        for i in range(5):
            # 对数据进行平滑滤波和剔除初始值处理
            waveform_data = self.waveforms_buffer[i]
            if len(waveform_data) > 10:  # 确保数据足够长
                # 1. 平滑滤波
                waveform_data = smooth_data(waveform_data, window_size=10)
                # 2. 剔除初始值（将数据减去第一个值）
                if len(waveform_data) > 0:
                    initial_value = waveform_data[0]
                else:
                    initial_value = 0

                waveform_data = [x - initial_value for x in waveform_data]

            # 更新波形图
            self.update_waveform(waveform_data, i)

    def check_data_length(self):
        """检查数据长度是否达到 224，如果达到则触发保存和推理"""
        if self.is_inferring:  # 如果正在推理，不触发新的推理
            return

        if all(len(data) >= self.buffer_size for data in self.waveforms_buffer):  # 检查所有通道的数据长度是否足够
            self.save_fused_images()  # 保存融合图像并触发推理

    def update_waveform(self, waveform_data, plot_index):
        """更新波形图"""
        if plot_index >= len(self.plot_widgets):  # 确保 plot_index 在范围内
            return

        plot_widget = self.plot_widgets[plot_index]
        plot_widget.clear()

        # 为每个通道指定颜色和加粗的线条
        colors = ['#87CEFA', '#98FB98', '#DDA0DD', '#FFB6C1', '#FFFACD']  # 浅蓝色、浅绿色、淡紫色、柔粉色、浅黄色
        plot_widget.plot(waveform_data, pen=pg.mkPen(color=colors[plot_index], width=2))  # 颜色和线条宽度

        # 检查 waveform_data 的长度，确保 image_size 不大于数据长度
        n_timestamps = len(waveform_data)
        if n_timestamps < 224:
            # 如果数据不足 224，填充零
            waveform_data = np.pad(waveform_data, (0, 224 - n_timestamps), mode='constant')
        elif n_timestamps > 224:
            # 如果数据超过 224，截断
            waveform_data = waveform_data[:224]

        # 使用固定的 image_size (224)
        gasf = GramianAngularField(image_size=224, method='summation')
        gadf = GramianAngularField(image_size=224, method='difference')

        # 转换为 GASF 和 GADF 图像（原始矩阵）
        gasf_image = gasf.fit_transform([waveform_data])[0]
        self.gasf_images[plot_index] = gasf_image

        gadf_image = gadf.fit_transform([waveform_data])[0]
        self.gadf_images[plot_index] = gadf_image

        # 更新小画布中的 GAF 图像
        self.update_image_display(gasf_image, self.gasf_canvas_widgets[plot_index])
        self.update_image_display(gadf_image, self.gadf_canvas_widgets[plot_index])

        # 融合 GAF 图像并更新大画布
        self.fuse_images(self.gasf_images, self.right_large_canvas)
        self.fuse_images(self.gadf_images, self.left_large_canvas)

    def update_image_display(self, image, canvas_widget):
        """更新图像显示"""
        # 使用 jet 调色板将矩阵转换为彩色图像
        colored_image = plt.cm.jet(image)  # 使用 jet 调色板
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  # 转换为 0-255 的 RGB 图像

        height, width, _ = colored_image.shape
        qimage = QImage(colored_image.data, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # 获取画布的宽度和高度
        canvas_width = canvas_widget.width()
        canvas_height = canvas_widget.height()

        # 按比例缩放图像
        scaled_pixmap = pixmap.scaled(canvas_width, canvas_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 清除画布并添加缩放后的图像
        canvas_widget.scene().clear()
        canvas_widget.scene().addPixmap(scaled_pixmap)

    def fuse_images(self, images, large_canvas, save=True):
        """融合图像并显示"""
        # 将5张GAF图像的原始矩阵相加
        fused_image = np.sum(images, axis=0)

        # 归一化融合后的矩阵（可选，避免像素值超出范围）
        fused_image = (fused_image - np.min(fused_image)) / (np.max(fused_image) - np.min(fused_image))

        # 将融合后的矩阵转换为彩色图像并显示
        self.update_image_display(fused_image, large_canvas)

        # 返回融合后的图像矩阵
        return fused_image

    def save_fused_images(self):
        """保存融合后的GASF和GADF图像"""
        if self.is_inferring:  # 如果正在推理，不重复触发
            return

        self.is_inferring = True  # 标记为正在推理

        # 保存GASF图像
        gasf_fused_image = self.fuse_images(self.gasf_images, self.right_large_canvas, save=False)
        print(f"GASF fused image shape: {gasf_fused_image.shape}")  # 添加打印语句
        gasf_fused_path = "fused_gasf.png"
        self.save_image(gasf_fused_image, gasf_fused_path)

        # 保存GADF图像
        gadf_fused_image = self.fuse_images(self.gadf_images, self.left_large_canvas, save=False)
        print(f"GADF fused image shape: {gadf_fused_image.shape}")  # 添加打印语句
        gadf_fused_path = "fused_gadf.png"
        self.save_image(gadf_fused_image, gadf_fused_path)

        # 将图像传递给神经网络进行推理
        self.infer_neural_network(gasf_fused_path, gadf_fused_path)

        print("融合图像已保存并触发推理！")

    def save_image(self, image, filename):
        """将图像保存为文件"""
        # 使用 matplotlib 的 jet 调色板将矩阵转换为彩色图像
        colored_image = plt.cm.jet(image)  # 使用 jet 调色板
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  # 转换为 0-255 的 RGB 图像

        # 保存图像
        plt.imsave(filename, colored_image)
        print(f"Image saved as {filename}")  # 添加打印语句

    def infer_neural_network(self, gasf_fused_path, gadf_fused_path):
        """神经网络推理逻辑"""
        try:
            # 读取 PNG 图像
            gasf_fused_image = plt.imread(gasf_fused_path)  # 读取 RGB 或 RGBA 图像
            gadf_fused_image = plt.imread(gadf_fused_path)

            # 确保是 RGB 格式（去掉 Alpha 通道，如果有的话）
            if gasf_fused_image.shape[-1] == 4:  # 如果是 RGBA
                gasf_fused_image = gasf_fused_image[:, :, :3]
            if gadf_fused_image.shape[-1] == 4:  # 如果是 RGBA
                gadf_fused_image = gadf_fused_image[:, :, :3]

            # 将图像转换为 Tensor 并调整通道顺序 (C, H, W)
            gasf_tensor = torch.from_numpy(gasf_fused_image).permute(2, 0, 1).unsqueeze(0).to(device)
            gadf_tensor = torch.from_numpy(gadf_fused_image).permute(2, 0, 1).unsqueeze(0).to(device)

            print(f"GASF tensor shape: {gasf_tensor.shape}")  # 应该是 (1, 3, H, W)
            print(f"GADF tensor shape: {gadf_tensor.shape}")

            # 推理
            with torch.no_grad():
                outputs = self.model(gasf_tensor, gadf_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_class = predicted.item()

                # 创建映射字典
                class_mapping = {
                    0: "Finger heart",
                    1: "Fist bump",
                    2: "OK",
                    3: "Thumbs up",
                    4: "",

                }

                # 获取映射后的输出
                mapped_output = class_mapping.get(predicted_class, "未知类别")
                print(f"Predicted class: {predicted_class} => 输出: {mapped_output}")

                # 将 predicted_class 通过信号发送给 WebSocket 线程
                self.websocket_thread.predicted_class_signal.emit(mapped_output)

                # 根据识别结果设置图片
                self.update_images_based_on_prediction(predicted_class)

        except Exception as e:
            print(f"Error in infer_neural_network: {e}")
        finally:
            # 推理完成后清空所有缓冲区，并解除推理锁定
            self.waveforms_buffer = [[] for _ in range(5)]  # 清空波形缓冲区
            self.waveforms_data = [[] for _ in range(5)]   # 清空波形数据
            self.gasf_images = [np.zeros((224, 224)) for _ in range(5)]  # 清空 GASF 图像
            self.gadf_images = [np.zeros((224, 224)) for _ in range(5)]  # 清空 GADF 图像
            self.is_inferring = False  # 解除推理锁定

    def update_images_based_on_prediction(self, predicted_class):
        """根据推理结果更新显示的图片"""
        new_size = self.size()
        width_scale = new_size.width() / self.original_size.width()
        height_scale = new_size.height() / self.original_size.height()

        # 根据识别结果设置图片
        if predicted_class == 0:
            self.left_image_label.setPixmap(
                QPixmap("图3.png").scaled(int(400 * width_scale), int(200 * height_scale), Qt.KeepAspectRatio))
            self.right_image_label.setPixmap(
                QPixmap("图33.png").scaled(int(400 * width_scale), int(200 * height_scale), Qt.KeepAspectRatio))
        elif predicted_class == 1:
            self.left_image_label.setPixmap(
                QPixmap("图4.png").scaled(int(400 * width_scale), int(200 * height_scale), Qt.KeepAspectRatio))
            self.right_image_label.setPixmap(
                QPixmap("图44.png").scaled(int(400 * width_scale), int(200 * height_scale), Qt.KeepAspectRatio))
        elif predicted_class == 2:
            self.left_image_label.setPixmap(
                QPixmap("图1.png").scaled(int(400 * width_scale), int(200 * height_scale), Qt.KeepAspectRatio))
            self.right_image_label.setPixmap(
                QPixmap("图11.png").scaled(int(400 * width_scale), int(200 * height_scale), Qt.KeepAspectRatio))
        elif predicted_class == 3:
            self.left_image_label.setPixmap(
                QPixmap("图2.png").scaled(int(400 * width_scale), int(200 * height_scale), Qt.KeepAspectRatio))
            self.right_image_label.setPixmap(
                QPixmap("图22.png").scaled(int(400 * width_scale), int(200 * height_scale), Qt.KeepAspectRatio))
        else:
            # 默认图片（如果识别结果超出范围）
            self.left_image_label.setPixmap(
                QPixmap("9.png").scaled(int(400 * width_scale), int(200 * height_scale), Qt.KeepAspectRatio))
            self.right_image_label.setPixmap(
                QPixmap("10.png").scaled(int(400 * width_scale), int(200 * height_scale), Qt.KeepAspectRatio))


def closeEvent(self, event):
        """关闭事件处理"""
        self.serial_thread.quit()
        self.serial_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())