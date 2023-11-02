import os
import cv2
import paddle
import argparse
import numpy as np

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|行人检测|')
parser.add_argument('--image_path', default='image/demo.jpg', type=str, help='|图片位置|')
parser.add_argument('--model_path', default='model', type=str, help='|模型位置|')
parser.add_argument('--device', default='cpu', type=str, help='|设备|')
parser.add_argument('--threshold', default=0.8, type=float, help='|行人识别阈值，0.8为基准|')
parser.add_argument('--reid', default=True, type=bool, help='|是否对检测到的行人进行特征提取和匹配，返回人物编号0、1、2...|')
parser.add_argument('--reid_model_path', default='reid_model', type=str, help='|行人特征提取模型位置|')
parser.add_argument('--feature_database', default='feature_database.txt', type=str, help='|行人特征数据库|')
parser.add_argument('--feature_add', default=True, type=bool, help='|当行人特征数据库中没有此人时，是否增加其特征(暂时的)，False时不匹配返回None|')
parser.add_argument('--reid_threshold', default=0.85, type=float, help='|特征匹配阈值，0.85为基准|')
parser.add_argument('--draw', default=True, type=bool, help='|测试时可以启用画图|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()
args.device = 'cuda' if args.device.lower() in ['gpu', 'cuda'] else 'cpu'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
class recognition_class:
    def __init__(self, args):
        self.threshold = args.threshold
        infer_model = os.path.join(args.model_path, 'model.pdmodel')
        infer_params = os.path.join(args.model_path, 'model.pdiparams')
        config = paddle.inference.Config(infer_model, infer_params)
        if args.model_path == 'cuda':
            config.enable_use_gpu(200, 0)  # 初始化200M显存，使用gpu id为0
            config.switch_ir_optim(True)  # 开启IR优化
        else:
            config.disable_gpu()  # 使用cpu
            config.set_cpu_math_library_num_threads(1)  # 设置cpu线程数
        config.disable_glog_info()  # 推理时不会输出log信息
        config.enable_memory_optim()  # 可以分享内存
        config.switch_use_feed_fetch_ops(False)  # disable feed, fetch OP, needed by zero_copy_run
        self.model = paddle.inference.create_predictor(config)
        self.input_image = self.model.get_input_handle('image')  # 输入接口
        self.input_scale_factor = self.model.get_input_handle('scale_factor')

    def _image_deal(self, image):
        h, w, _ = image.shape
        image = cv2.resize(image, (640, 640)).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        scale_factor = np.array([640 / h, 640 / w])[np.newaxis].astype(np.float32)
        return image, scale_factor

    def _screen(self, pred):
        judge = np.where(pred[:, 1] > self.threshold, True, False)
        pred = pred[judge]
        return pred

    def predict(self, image):  # image:RGB
        image_resize, scale_factor = self._image_deal(image)  # 图片处理
        self.input_image.copy_from_cpu(image_resize)  # 传入数据
        self.input_scale_factor.copy_from_cpu(scale_factor)  # 传入数据
        self.model.run()  # 模型推理
        output_name = self.model.get_output_names()  # 输出信息
        output_tensor = self.model.get_output_handle(output_name[0])  # 输出结果
        pred = output_tensor.copy_to_cpu()  # array([[类别，置信度，x1,y1,x2,y2],...]) 已经非极大值抑制和从高到低排好序
        pred_screen = self._screen(pred[0:10])  # 不超过10个人
        return pred_screen[:, 2:6].astype(np.int32)


class reid_class:
    def __init__(self, args):
        infer_model = os.path.join(args.reid_model_path, 'model.pdmodel')
        infer_params = os.path.join(args.reid_model_path, 'model.pdiparams')
        config = paddle.inference.Config(infer_model, infer_params)
        if args.model_path == 'cuda':
            config.enable_use_gpu(200, 0)  # 初始化200M显存，使用gpu id为0
            config.switch_ir_optim(True)  # 开启IR优化
        else:
            config.disable_gpu()  # 使用cpu
            config.set_cpu_math_library_num_threads(1)  # 设置cpu线程数
        config.disable_glog_info()  # 推理时不会输出log信息
        config.enable_memory_optim()  # 可以分享内存
        config.switch_use_feed_fetch_ops(False)  # disable feed, fetch OP, needed by zero_copy_run
        self.model = paddle.inference.create_predictor(config)
        self.input_image = self.model.get_input_handle('x')  # 输入接口
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _image_deal(self, image):
        image = (image / 255 - self.mean) / self.std
        image = image.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        return image

    def predict(self, image):  # image:RGB
        image = self._image_deal(image)  # 图片处理
        self.input_image.copy_from_cpu(image)  # 传入数据
        self.model.run()  # 模型推理
        output_name = self.model.get_output_names()  # 输出信息
        output_tensor = self.model.get_output_handle(output_name[0])  # 输出结果
        feature = output_tensor.copy_to_cpu()  # array([[类别，置信度，x1,y1,x2,y2],...]) 已经非极大值抑制和从高到低排好序
        feature /= np.linalg.norm(feature, axis=1)
        return feature  # np.array(batch, 256)(np.float32)


class detection_class:
    def __init__(self, args):
        self.model1 = recognition_class(args)
        if args.reid:
            self.model2 = reid_class(args)
            self.reid = args.reid
            self.reid_threshold = args.reid_threshold
            self.feature_add = args.feature_add
            if os.path.exists(args.feature_database):
                with open(args.feature_database) as f:
                    line_all = [_.strip() for _ in f.readlines()]
                self.feature_database = np.array(line_all) if line_all else 'None'
            else:
                self.feature_database = 'None'

    def _cut_image(self, image, box_all):
        image_list = []
        for box in box_all:
            image_list.append(image[box[1]:box[3], box[0]:box[2]].copy())
        return image_list

    def _feature_match(self, feature):
        if self.feature_database == 'None':  # 没有特征数据库
            if self.feature_add:  # 添加到特征库中
                self.feature_database = feature
                id = 0
            else:  # 没有此人
                id = None
        else:
            similarity = np.dot(self.feature_database, feature.T)
            max_value = np.max(similarity)
            if max_value > self.reid_threshold:
                id = np.argmax(similarity)
            elif self.feature_add:  # 添加到特征库中
                self.feature_database = np.concatenate([feature, self.feature_database], axis=0)
                id = len(self.feature_database) - 1
            else:  # 没有此人
                id = None
        return id

    def draw(self, image, box_all, id_list):
        for (x1, y1, x2, y2), id in zip(box_all, id_list):
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(image, f'id:{id}', (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return image

    def predict(self, image):  # image:RGB
        # 行人检测
        box_list = self.model1.predict(image).astype(np.int32).tolist()
        # 行人身份识别
        id_list = [None for _ in range(len(box_list))]  # 最小值从0开始
        if self.reid:
            image_list = self._cut_image(image, box_list)  # 裁剪图片
            for i, image_cut in enumerate(image_list):
                feature = self.model2.predict(image_cut)
                id_list[i] = self._feature_match(feature)
        return box_list, id_list


if __name__ == '__main__':
    image = cv2.imdecode(np.fromfile(args.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = detection_class(args)
    box_list, id_list = model.predict(image)
    # 画图
    if args.draw:
        image_draw = model.draw(image, box_list, id_list)  # image和image_draw共用内存
        image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'save.jpg', image_draw)
    print(box_list, id_list)
