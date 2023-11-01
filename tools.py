"""
    从一段视频中每隔frame_interval帧抽取一帧
"""
from moviepy.editor import VideoFileClip
import imageio
import os
from PIL import Image

# 视频文件路径
video_file = 'data.mp4'
# 保存帧图像的文件夹
output_folder = 'mydata'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 打开视频文件
clip = VideoFileClip(video_file)

frame_interval = 30  # 每隔frame_interval帧抽取一帧
desired_width = 960
desired_height = 540

frame_count = 0
frame_index = 1  # 从 'frame_0001.jpg' 开始

for frame in clip.iter_frames(fps=clip.fps):
    frame_count += 1
    if frame_count % frame_interval == 0:
        # 调整帧图像大小为576x768
        frame_image = Image.fromarray(frame)
        frame_image = frame_image.resize((desired_width, desired_height))

        # 保存帧图像为JPG文件
        frame_filename = os.path.join(output_folder, f'frame_{frame_index:04d}.jpg')
        frame_image.save(frame_filename)
        frame_index += 1

print(
    f'成功抽取了 {frame_count // frame_interval} 帧图像，并保存在 {output_folder} 文件夹中，并调整大小为 {desired_width}x{desired_height} 像素。')
