# 导入必要的库
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
import os
import numpy as np
import random
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from lime import lime_image
import shap
from skimage.segmentation import quickshift
from skimage import exposure
import time
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")

# 设置随机种子以确保可重复性
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# 定义数据路径
train_path = "/Users/panteng/Senimararbeit_XAI/Covid19_XAI/data/Covid19-dataset/train"  # 训练集路径
test_path = "/Users/panteng/Senimararbeit_XAI/Covid19_XAI/data/Covid19-dataset/test"  # 测试集路径

# 定义图像大小和批次大小
img_height, img_width = 224, 224  # 使用较大的图像尺寸以适应预训练模型
batch_size = 32

# 定义数据增强和预处理（使用预训练模型的预处理函数）
# 训练数据生成器，包含数据增强
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # 预处理函数
    rotation_range=15,  # 随机旋转角度
    width_shift_range=0.1,  # 水平平移
    height_shift_range=0.1,  # 垂直平移
    zoom_range=0.1,  # 随机缩放
    horizontal_flip=True,  # 随机水平翻转
    validation_split=0.2,  # 划分 20% 的数据作为验证集
)

# 测试数据生成器，仅进行预处理，不进行数据增强
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# 生成训练数据
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",  # 使用训练集
    shuffle=True,  # 打乱数据
    seed=seed,
)

# 生成验证数据
validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",  # 使用验证集
    shuffle=False,
    seed=seed,
)

# 生成测试数据
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=1,  # 测试时 batch_size 设为 1
    class_mode="categorical",
    shuffle=False,  # 不打乱数据，方便后续预测和评估
)

# 获取类别索引与类别名称的映射
class_indices = train_generator.class_indices
print("类别索引与名称的映射：", class_indices)


# 定义模型构建函数
def build_model(input_shape, num_classes):
    """
    构建卷积神经网络模型。

    参数：
    - input_shape: 输入图像的形状
    - num_classes: 类别数量

    返回：
    - model: 构建的 Keras 模型
    """
    # 使用预训练的 VGG16 模型，不包含顶层的全连接层
    base_model = tf.keras.applications.VGG16(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # 冻结预训练模型的所有层
    for layer in base_model.layers:
        layer.trainable = False

    # 构建自定义的顶层
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # 构建完整的模型
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


# 获取输入形状和类别数量
input_shape = (img_height, img_width, 3)
num_classes = train_generator.num_classes

# 构建模型
model = build_model(input_shape, num_classes)

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # 使用 Adam 优化器
    loss="categorical_crossentropy",  # 多分类的损失函数
    metrics=["accuracy"],  # 评估指标
)

# 打印模型结构
model.summary()

# 定义回调函数
early_stopping = EarlyStopping(
    monitor="val_loss",  # 监控验证集的损失
    patience=5,  # 若连续 5 个周期没有提升，则停止训练
    restore_best_weights=True,  # 恢复到最佳的模型权重
)

model_checkpoint = ModelCheckpoint(
    "best_model.keras",  # 保存模型的文件名
    monitor="val_loss",  # 监控验证集的损失
    save_best_only=True,  # 仅保存性能最好的模型
    verbose=1,
    save_weights_only=False,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",  # 监控验证集的损失
    factor=0.5,  # 当指标不提升时，学习率减少的因子
    patience=3,  # 等待 3 个周期
    min_lr=1e-6,  # 学习率的下限
    verbose=1,
)

# 训练模型
history = model.fit(
    train_generator,
    epochs=50,  # 最大训练周期
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1,
)


# 绘制训练和验证的准确率和损失曲线
def plot_history(history):
    """
    绘制训练过程的准确率和损失曲线。

    参数：
    - history: 训练过程中记录的指标
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, "bo-", label="Training Accuracy")
    plt.plot(epochs_range, val_acc, "r*-", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, "bo-", label="Training Loss")
    plt.plot(epochs_range, val_loss, "r*-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.show()


# 调用绘图函数
plot_history(history)

# 加载最佳模型
model.load_weights("best_model.keras")

# 在测试集上进行预测
test_steps = test_generator.samples  # 测试集的样本数量
predictions = model.predict(test_generator, steps=test_steps, verbose=1)

# 获取预测的类别
y_pred = np.argmax(predictions, axis=1)

# 获取真实的类别
y_true = test_generator.classes

# 打印分类报告
report = classification_report(y_true, y_pred, target_names=list(class_indices.keys()))
print("分类报告：\n", report)

# 绘制混淆矩阵
confusion_mtx = confusion_matrix(y_true, y_pred)  # 修改变量名，避免与 cm 模块冲突
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_mtx,
    annot=True,
    fmt="d",
    xticklabels=list(class_indices.keys()),
    yticklabels=list(class_indices.keys()),
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------------------
# LIME 解释部分
# -----------------------------------------

# 创建 LIME 图像解释器
explainer = lime_image.LimeImageExplainer()


# 定义预测函数，优化批量预测
def predict_fn(images):
    images = np.array(images)
    # 将像素值从 [0, 1] 恢复到 [0, 255]
    images = images * 255.0
    # 确保数据类型为 float32，以匹配模型的输入要求
    images = images.astype(np.float32)
    images = preprocess_input(images)
    # 批量预测，设置合适的 batch_size
    preds = model.predict(images, batch_size=32, verbose=0)
    return preds


# 自定义超像素分割函数，针对肺部 X 光片优化
def segmentation_fn(image):
    # 图像像素值在 [0, 1] 范围内
    # 对图像进行对比度增强
    image = exposure.equalize_adapthist(image, clip_limit=0.03)
    # 使用 quickshift 算法进行超像素分割
    segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
    return segments


# 获取测试集中的所有图像文件路径和对应的标签
test_image_paths = []
test_labels = []
for root, dirs, files in os.walk(test_path):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(root, filename)
            test_image_paths.append(filepath)
            # 获取对应的标签
            label = os.path.basename(root)
            test_labels.append(label)

# 创建类别名称和索引的映射
label_to_index = {v: k for k, v in test_generator.class_indices.items()}

# 随机选择几张测试图像进行解释
num_images_to_explain = 3  # 选择要解释的图像数量
selected_indices = random.sample(range(len(test_image_paths)), num_images_to_explain)

for idx in selected_indices:
    img_path = test_image_paths[idx]
    true_label = test_labels[idx]

    # 加载原始灰度图像
    test_image = load_img(
        img_path, target_size=(img_height, img_width), color_mode="grayscale"
    )
    test_image_array = img_to_array(test_image)

    # 将灰度图像转换为 RGB（重复灰度值到 R、G、B 三个通道）
    test_image_array_rgb = np.repeat(test_image_array, 3, axis=2)

    # 将像素值归一化到 [0, 1]
    test_image_array_rgb = test_image_array_rgb.astype(np.float32) / 255.0

    # 扩展维度以符合模型输入
    test_image_expanded = np.expand_dims(test_image_array_rgb, axis=0)

    # 预测图像类别
    preds = predict_fn(test_image_expanded)
    pred_label_index = np.argmax(preds)
    pred_label = list(class_indices.keys())[pred_label_index]

    print(f"图像路径: {img_path}")
    print(f"真实标签: {true_label}, 预测标签: {pred_label}")

    # 记录开始时间
    start_time = time.time()

    # 使用 LIME 进行解释，指定自定义的分割函数
    explanation = explainer.explain_instance(
        test_image_array_rgb,
        predict_fn,
        top_labels=num_classes,  # 确保包含所有类别
        hide_color=0,
        num_samples=2000,  # 保持原有的扰动样本数量
        batch_size=32,  # 设置批量大小，提高预测效率
        segmentation_fn=segmentation_fn,  # 使用自定义的超像素分割函数
    )

    # 输出解释所花费的时间
    elapsed_time = time.time() - start_time
    print(f"LIME 解释完成，耗时 {elapsed_time:.2f} 秒")

    # 获取解释结果，增加 num_features
    temp, mask = explanation.get_image_and_mask(
        label=pred_label_index,
        positive_only=False,
        hide_rest=False,
        num_features=15,  # 增加超像素数量，获得更详细的解释
        min_weight=0.0,
    )

    # 确保 mask 是整数类型
    mask = mask.astype(int)

    # 将 dict_heatmap 的键转换为整数
    dict_heatmap = dict(explanation.local_exp[pred_label_index])
    dict_heatmap = {int(k): v for k, v in dict_heatmap.items()}

    # 获取 mask 中的唯一值
    mask_indices = np.unique(mask)

    # 为缺失的索引赋默认值 0.0
    for idx_key in mask_indices:
        if idx_key not in dict_heatmap:
            dict_heatmap[idx_key] = 0.0

    # 定义安全的 get 函数
    def safe_get(key):
        return dict_heatmap.get(int(key), 0.0)  # 未找到的键返回 0.0

    # 获取热力图
    heatmap = np.vectorize(safe_get)(mask)
    heatmap = heatmap.astype(float)

    # 检查并替换 nan 值
    if np.isnan(heatmap).any():
        print("Heatmap contains NaNs, replacing with zeros.")
        heatmap = np.nan_to_num(heatmap)

    # 归一化热力图
    if np.max(np.abs(heatmap)) > 0:
        heatmap = heatmap / np.max(np.abs(heatmap))
    else:
        print("Heatmap contains only zero values.")

    # 可视化原始图像和热力图
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # 原始灰度图像
    ax[0].imshow(test_image_array.squeeze(), cmap="gray")
    ax[0].axis("off")
    ax[0].set_title(
        f"Original Image\nTrue Label: {true_label}\nPredicted Label: {pred_label}"
    )

    # 对比度增强后的图像
    enhanced_image = exposure.equalize_adapthist(
        test_image_array.squeeze() / 255.0, clip_limit=0.03
    )
    ax[1].imshow(enhanced_image, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("Contrast Enhanced Image")

    # 叠加热力图
    ax[2].imshow(enhanced_image, cmap="gray", interpolation="nearest")
    cmap_seismic = cm.seismic  # 使用 seismic 颜色映射
    heatmap_display = ax[2].imshow(
        heatmap, cmap=cmap_seismic, alpha=0.5, interpolation="nearest"
    )
    ax[2].axis("off")
    ax[2].set_title("LIME Explanation (Heatmap)")

    # 添加颜色条
    cbar = fig.colorbar(heatmap_display, ax=ax[2], fraction=0.046, pad=0.04)
    cbar.set_label("Contribution to Prediction", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

    # -----------------------------------------

# -----------------------------------------
# SHAP 解释部分
# -----------------------------------------


# 定义预测函数，优化批量预测
def predict_fn(images):
    images = np.array(images)
    # 将像素值从 [0, 1] 恢复到 [0, 255]
    images = images * 255.0
    # 确保数据类型为 float32，以匹配模型的输入要求
    images = images.astype(np.float32)
    images = preprocess_input(images)
    # 批量预测，设置合适的 batch_size
    preds = model.predict(images, batch_size=32, verbose=0)
    return preds


# 从 train_generator 中提取一批数据并将其转换为 NumPy 数组作为背景数据
x_train_sample, y_train_sample = [], []

# 手动迭代从 train_generator 提取数据
for i in range(len(train_generator)):
    x_batch, y_batch = next(train_generator)
    x_train_sample.append(x_batch)
    y_train_sample.append(y_batch)
    if len(x_train_sample) >= 100:  # 假设我们需要 100 张图像作为背景数据
        break

# 将提取的数据转换为 NumPy 数组
x_train_sample = np.concatenate(x_train_sample, axis=0)
y_train_sample = np.concatenate(y_train_sample, axis=0)

# 确保背景数据是 NumPy 数组
background_data = x_train_sample[:100]  # 选择前 100 张图像

# 创建 SHAP DeepExplainer，使用部分训练数据作为背景
explainer_shap = shap.DeepExplainer(model, background_data)


# 自定义超像素分割函数
def segmentation_fn(image):
    image = exposure.equalize_adapthist(image, clip_limit=0.03)
    segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
    return segments


# 获取测试集中的所有图像文件路径和对应的标签
test_image_paths = []
test_labels = []
for root, dirs, files in os.walk(test_path):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(root, filename)
            test_image_paths.append(filepath)
            # 获取对应的标签
            label = os.path.basename(root)
            test_labels.append(label)

# 随机选择几张测试图像进行解释
num_images_to_explain = 3  # 选择要解释的图像数量
selected_indices = random.sample(range(len(test_image_paths)), num_images_to_explain)

# 创建类别名称和索引的映射
class_indices = test_generator.class_indices
label_to_index = {v: k for k, v in class_indices.items()}

for idx in selected_indices:
    img_path = test_image_paths[idx]
    true_label = test_labels[idx]

    # 加载原始灰度图像
    test_image = load_img(
        img_path, target_size=(img_height, img_width), color_mode="grayscale"
    )
    test_image_array = img_to_array(test_image)

    # 将灰度图像转换为 RGB（重复灰度值到 R、G、B 三个通道）
    test_image_array_rgb = np.repeat(test_image_array, 3, axis=2)

    # 将像素值归一化到 [0, 1]
    test_image_array_rgb = test_image_array_rgb.astype(np.float32) / 255.0

    # 扩展维度以符合模型输入
    test_image_expanded = np.expand_dims(test_image_array_rgb, axis=0)

    # 预测图像类别
    preds = predict_fn(test_image_expanded)
    pred_label_index = np.argmax(preds)
    pred_label = list(class_indices.keys())[pred_label_index]

    print(f"图像路径: {img_path}")
    print(f"真实标签: {true_label}, 预测标签: {pred_label}")

    # 记录开始时间
    start_time = time.time()

    # 获取 SHAP 值
    shap_values = explainer_shap.shap_values(test_image_expanded)

    # 输出解释所花费的时间
    elapsed_time = time.time() - start_time
    print(f"SHAP 解释完成，耗时 {elapsed_time:.2f} 秒")

    # 超像素分割
    segments = segmentation_fn(test_image_array_rgb)

    # 获取 SHAP 值的维度并对通道进行平均
    mean_shap_values = np.mean(
        shap_values[0], axis=-1
    )  # 对每个通道的 SHAP 值进行平均，得到 224x224

    # 确保 `mean_shap_values` 和 `segments` 一致
    mask = np.zeros(segments.shape)  # 保证 mask 是 224x224

    # 遍历每个超像素区域，将对应的 SHAP 值应用到该区域
    for i in np.unique(segments):
        mask[segments == i] = mean_shap_values[segments == i].mean()

    # 可视化 SHAP 解释结果
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # 原始灰度图像
    ax[0].imshow(test_image_array.squeeze(), cmap="gray")
    ax[0].axis("off")
    ax[0].set_title(
        f"Original Image\nTrue Label: {true_label}\nPredicted Label: {pred_label}"
    )

    # 对比度增强后的图像
    enhanced_image = exposure.equalize_adapthist(
        test_image_array.squeeze() / 255.0, clip_limit=0.03
    )
    ax[1].imshow(enhanced_image, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("Contrast Enhanced Image")

    # 叠加超像素的 SHAP 解释图
    ax[2].imshow(enhanced_image, cmap="gray", interpolation="nearest")
    cmap_seismic = cm.seismic  # 使用 seismic 颜色映射
    heatmap_display = ax[2].imshow(
        mask, cmap=cmap_seismic, alpha=0.5, interpolation="nearest"
    )
    ax[2].axis("off")
    ax[2].set_title("SHAP Explanation (Heatmap)")

    # 添加颜色条
    cbar = fig.colorbar(heatmap_display, ax=ax[2], fraction=0.046, pad=0.04)
    cbar.set_label("SHAP Values", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()


# LIME 解释部分
# -----------------------------------------

# 创建 LIME 图像解释器
explainer = lime_image.LimeImageExplainer()


# 定义预测函数，优化批量预测
def predict_fn(images):
    images = np.array(images)
    # 将像素值从 [0, 1] 恢复到 [0, 255]
    images = images * 255.0
    # 确保数据类型为 float32，以匹配模型的输入要求
    images = images.astype(np.float32)
    images = preprocess_input(images)
    # 批量预测，设置合适的 batch_size
    preds = model.predict(images, batch_size=32, verbose=0)
    return preds


# 自定义超像素分割函数，针对肺部 X 光片优化
def segmentation_fn(image):
    # 图像像素值在 [0, 1] 范围内
    # 对图像进行对比度增强
    image = exposure.equalize_adapthist(image, clip_limit=0.03)
    # 使用 quickshift 算法进行超像素分割
    segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
    return segments


# 获取测试集中的所有图像文件路径和对应的标签
test_image_paths = []
test_labels = []
for root, dirs, files in os.walk(test_path):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(root, filename)
            test_image_paths.append(filepath)
            # 获取对应的标签
            label = os.path.basename(root)
            test_labels.append(label)

# 创建类别名称和索引的映射
label_to_index = {v: k for k, v in test_generator.class_indices.items()}

# 遍历测试集中的所有图像并进行解释
for img_path, true_label in zip(test_image_paths, test_labels):
    # 加载原始灰度图像
    test_image = load_img(
        img_path, target_size=(img_height, img_width), color_mode="grayscale"
    )
    test_image_array = img_to_array(test_image)

    # 将灰度图像转换为 RGB（重复灰度值到 R、G、B 三个通道）
    test_image_array_rgb = np.repeat(test_image_array, 3, axis=2)

    # 将像素值归一化到 [0, 1]
    test_image_array_rgb = test_image_array_rgb.astype(np.float32) / 255.0

    # 扩展维度以符合模型输入
    test_image_expanded = np.expand_dims(test_image_array_rgb, axis=0)

    # 预测图像类别
    preds = predict_fn(test_image_expanded)
    pred_label_index = np.argmax(preds)
    pred_label = list(class_indices.keys())[pred_label_index]

    print(f"图像路径: {img_path}")
    print(f"真实标签: {true_label}, 预测标签: {pred_label}")

    # 记录开始时间
    start_time = time.time()

    # 使用 LIME 进行解释，指定自定义的分割函数
    explanation = explainer.explain_instance(
        test_image_array_rgb,
        predict_fn,
        top_labels=num_classes,  # 确保包含所有类别
        hide_color=0,
        num_samples=2000,  # 保持原有的扰动样本数量
        batch_size=32,  # 设置批量大小，提高预测效率
        segmentation_fn=segmentation_fn,  # 使用自定义的超像素分割函数
    )

    # 输出解释所花费的时间
    elapsed_time = time.time() - start_time
    print(f"LIME 解释完成，耗时 {elapsed_time:.2f} 秒")

    # 获取解释结果，增加 num_features
    temp, mask = explanation.get_image_and_mask(
        label=pred_label_index,
        positive_only=False,
        hide_rest=False,
        num_features=15,  # 增加超像素数量，获得更详细的解释
        min_weight=0.0,
    )

    # 确保 mask 是整数类型
    mask = mask.astype(int)

    # 将 dict_heatmap 的键转换为整数
    dict_heatmap = dict(explanation.local_exp[pred_label_index])
    dict_heatmap = {int(k): v for k, v in dict_heatmap.items()}

    # 获取 mask 中的唯一值
    mask_indices = np.unique(mask)

    # 为缺失的索引赋默认值 0.0
    for idx_key in mask_indices:
        if idx_key not in dict_heatmap:
            dict_heatmap[idx_key] = 0.0

    # 定义安全的 get 函数
    def safe_get(key):
        return dict_heatmap.get(int(key), 0.0)  # 未找到的键返回 0.0

    # 获取热力图
    heatmap = np.vectorize(safe_get)(mask)
    heatmap = heatmap.astype(float)

    # 检查并替换 nan 值
    if np.isnan(heatmap).any():
        print("Heatmap contains NaNs, replacing with zeros.")
        heatmap = np.nan_to_num(heatmap)

    # 归一化热力图
    if np.max(np.abs(heatmap)) > 0:
        heatmap = heatmap / np.max(np.abs(heatmap))
    else:
        print("Heatmap contains only zero values.")

    # 可视化原始图像和热力图
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # 原始灰度图像
    ax[0].imshow(test_image_array.squeeze(), cmap="gray")
    ax[0].axis("off")
    ax[0].set_title(
        f"Original Image\nTrue Label: {true_label}\nPredicted Label: {pred_label}"
    )

    # 对比度增强后的图像
    enhanced_image = exposure.equalize_adapthist(
        test_image_array.squeeze() / 255.0, clip_limit=0.03
    )
    ax[1].imshow(enhanced_image, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("Contrast Enhanced Image")

    # 叠加热力图
    ax[2].imshow(enhanced_image, cmap="gray", interpolation="nearest")
    cmap_seismic = cm.seismic  # 使用 seismic 颜色映射
    heatmap_display = ax[2].imshow(
        heatmap, cmap=cmap_seismic, alpha=0.5, interpolation="nearest"
    )
    ax[2].axis("off")
    ax[2].set_title("LIME Explanation (Heatmap)")

    # 添加颜色条
    cbar = fig.colorbar(heatmap_display, ax=ax[2], fraction=0.046, pad=0.04)
    cbar.set_label("Contribution to Prediction", rotation=270, labelpad=15)

    # 保存解释结果
    lime_save_path = os.path.join(
        "/Users/panteng/Senimararbeit_XAI/Covid19_XAI/reports/figures_LIME",
        f"{os.path.basename(img_path).split('.')[0]}_LIME.png",
    )
    plt.savefig(lime_save_path)
    plt.close(fig)

    print(f"LIME 解释结果已保存到: {lime_save_path}")


# SHAP 解释部分
# -----------------------------------------


# 定义预测函数，优化批量预测
def predict_fn(images):
    images = np.array(images)
    # 将像素值从 [0, 1] 恢复到 [0, 255]
    images = images * 255.0
    # 确保数据类型为 float32，以匹配模型的输入要求
    images = images.astype(np.float32)
    images = preprocess_input(images)
    # 批量预测，设置合适的 batch_size
    preds = model.predict(images, batch_size=32, verbose=0)
    return preds


# 从 train_generator 中提取一批数据并将其转换为 NumPy 数组作为背景数据
x_train_sample, y_train_sample = [], []

# 手动迭代从 train_generator 提取数据
for i in range(len(train_generator)):
    x_batch, y_batch = next(train_generator)
    x_train_sample.append(x_batch)
    y_train_sample.append(y_batch)
    if len(x_train_sample) >= 100:  # 假设我们需要 100 张图像作为背景数据
        break

# 将提取的数据转换为 NumPy 数组
x_train_sample = np.concatenate(x_train_sample, axis=0)
y_train_sample = np.concatenate(y_train_sample, axis=0)

# 确保背景数据是 NumPy 数组
background_data = x_train_sample[:100]  # 选择前 100 张图像

# 创建 SHAP DeepExplainer，使用部分训练数据作为背景
explainer_shap = shap.DeepExplainer(model, background_data)


# 自定义超像素分割函数
def segmentation_fn(image):
    image = exposure.equalize_adapthist(image, clip_limit=0.03)
    segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
    return segments


# 获取测试集中的所有图像文件路径和对应的标签
test_image_paths = []
test_labels = []
for root, dirs, files in os.walk(test_path):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(root, filename)
            test_image_paths.append(filepath)
            # 获取对应的标签
            label = os.path.basename(root)
            test_labels.append(label)

# 遍历所有测试集中的图像并进行解释
for img_path, true_label in zip(test_image_paths, test_labels):
    # 加载原始灰度图像
    test_image = load_img(
        img_path, target_size=(img_height, img_width), color_mode="grayscale"
    )
    test_image_array = img_to_array(test_image)

    # 将灰度图像转换为 RGB（重复灰度值到 R、G、B 三个通道）
    test_image_array_rgb = np.repeat(test_image_array, 3, axis=2)

    # 将像素值归一化到 [0, 1]
    test_image_array_rgb = test_image_array_rgb.astype(np.float32) / 255.0

    # 扩展维度以符合模型输入
    test_image_expanded = np.expand_dims(test_image_array_rgb, axis=0)

    # 预测图像类别
    preds = predict_fn(test_image_expanded)
    pred_label_index = np.argmax(preds)
    pred_label = list(class_indices.keys())[pred_label_index]

    print(f"图像路径: {img_path}")
    print(f"真实标签: {true_label}, 预测标签: {pred_label}")

    # 记录开始时间
    start_time = time.time()

    # 获取 SHAP 值
    shap_values = explainer_shap.shap_values(test_image_expanded)

    # 输出解释所花费的时间
    elapsed_time = time.time() - start_time
    print(f"SHAP 解释完成，耗时 {elapsed_time:.2f} 秒")

    # 超像素分割
    segments = segmentation_fn(test_image_array_rgb)

    # 获取 SHAP 值的维度并对通道进行平均
    mean_shap_values = np.mean(
        shap_values[0], axis=-1
    )  # 对每个通道的 SHAP 值进行平均，得到 224x224

    # 确保 `mean_shap_values` 和 `segments` 一致
    mask = np.zeros(segments.shape)  # 保证 mask 是 224x224

    # 遍历每个超像素区域，将对应的 SHAP 值应用到该区域
    for i in np.unique(segments):
        mask[segments == i] = mean_shap_values[segments == i].mean()

    # 可视化 SHAP 解释结果
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # 原始灰度图像
    ax[0].imshow(test_image_array.squeeze(), cmap="gray")
    ax[0].axis("off")
    ax[0].set_title(
        f"Original Image\nTrue Label: {true_label}\nPredicted Label: {pred_label}"
    )

    # 对比度增强后的图像
    enhanced_image = exposure.equalize_adapthist(
        test_image_array.squeeze() / 255.0, clip_limit=0.03
    )
    ax[1].imshow(enhanced_image, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("Contrast Enhanced Image")

    # 叠加超像素的 SHAP 解释图
    ax[2].imshow(enhanced_image, cmap="gray", interpolation="nearest")
    cmap_seismic = cm.seismic  # 使用 seismic 颜色映射
    heatmap_display = ax[2].imshow(
        mask, cmap=cmap_seismic, alpha=0.5, interpolation="nearest"
    )
    ax[2].axis("off")
    ax[2].set_title("SHAP Explanation (Heatmap)")

    # 添加颜色条
    cbar = fig.colorbar(heatmap_display, ax=ax[2], fraction=0.046, pad=0.04)
    cbar.set_label("SHAP Values", rotation=270, labelpad=15)

    # 保存解释结果
    shap_save_path = os.path.join(
        "/Users/panteng/Senimararbeit_XAI/Covid19_XAI/reports/figures_SHAP",
        f"{os.path.basename(img_path).split('.')[0]}_SHAP.png",
    )
    plt.savefig(shap_save_path)
    plt.close(fig)

    print(f"SHAP 解释结果已保存到: {shap_save_path}")
