import os
import shutil

from ultralytics import YOLO

model = YOLO("./runs/classify/train/weights/best.pt")  # load a custom model


def classify_images(image_path, model):
    # 进行图像分类
    results = model(image_path, show=False)

    # 获取分类结果（最可能的类别  置信度高的）
    res = results[0]
    top_class_id = res.probs.top1  # 获取置信度最高的类别id
    top_class_name = res.names[top_class_id]

    # 创建类别文件夹，如果不存在就创建
    class_folder = os.path.join("classified_images", top_class_name)
    os.makedirs(class_folder, exist_ok=True)

    # 获取原始图片的文件名
    pre_image_name = os.path.basename(image_path)

    # 目标路径
    destination_path = os.path.join(class_folder, pre_image_name)

    # 移动图片到目标文件夹
    shutil.move(image_path, destination_path)

    print(f"图片已保存到{destination_path}")
    return top_class_name


if __name__ == "__main__":
    # 需要分类的图片的路径
    target_path = "./flower_test"

    # 如果是文件夹，处理器中所有图片
    if os.path.isdir(target_path):
        # 获取文件夹中所有的图片
        image_extensions = [".jpg", ".jpeg", ".png"]
        for image_name in os.listdir(target_path):
            image_path = os.path.join(target_path, image_name)
            if os.path.isfile(image_path) and os.path.splitext(image_name)[1] in image_extensions:
                classify_images(image_path, model)
    else:
        classify_images(target_path, model)
