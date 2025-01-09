下载模型存放到 trained_checkpoint 目录  
把图像放入 input_images 目录  
运行 infer_limit_area.py 将会首先把 input_images_resize 目录中的图像移到回收站  
把 input_images 中的图像按比例缩放并存放到 input_images_resize  
对 input_images_resize 中的图像进行检测  
结果存放到 output_images  
