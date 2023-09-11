import os
import cv2
import paddle
import argparse
import numpy as np

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|行人检测|')
parser.add_argument('--model_path', default='model', type=str, help='|模型位置|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
parser.add_argument('--threshold', default=0.8, type=float, help='|阈值|')
args = parser.parse_args()
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
        config.disable_glog_info()  # disable print log when predict
        config.enable_memory_optim()  # enable shared memory
        config.switch_use_feed_fetch_ops(False)  # disable feed, fetch OP, needed by zero_copy_run
        self.model = paddle.inference.create_predictor(config)
        self.input_image = self.model.get_input_handle('image')  # 输入接口
        self.input_scale_factor = self.model.get_input_handle('scale_factor')

    def _image_deal(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        image = cv2.resize(image, (640, 640)).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        scale_factor = np.array([640 / h, 640 / w])[np.newaxis].astype(np.float32)
        return image, scale_factor

    def _screen(self, pred):
        judge = np.where(pred[:, 1] > self.threshold, True, False)
        pred = pred[judge]
        return pred

    def _draw(self, image, box_all):
        for x1, y1, x2, y2 in box_all.astype(np.int32):
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        return image

    def predict(self, image):  # image:BGR
        image_resize, scale_factor = self._image_deal(image)  # 图片处理
        self.input_image.copy_from_cpu(image_resize)  # 传入数据
        self.input_scale_factor.copy_from_cpu(scale_factor)  # 传入数据
        self.model.run()  # 模型推理
        output_name = self.model.get_output_names()  # 输出信息
        output_tensor = self.model.get_output_handle(output_name[0])  # 输出结果
        pred = output_tensor.copy_to_cpu()  # array([[类别，置信度，x1,y1,x2,y2],...]) 已经非极大值抑制和从高到低排好序
        pred_screen = self._screen(pred[0:10])  # 不超过10个人
        image_draw = self._draw(image, pred_screen[:, 2:6])  # image和image_draw共用内存是一样的
        return image_draw, pred_screen[:, 2:6]


class detection_class:
    def __init__(self, args):
        self.model = recognition_class(args)

    def predict(self, image):  # image:BGR
        image_draw, box_all = self.model.predict(image)
        cv2.imwrite('draw.jpg', image_draw)
        result = {'image_draw': image_draw, 'box_all': box_all.tolist()}
        return result


if __name__ == '__main__':
    image = cv2.imread('001.jpg')
    detection = detection_class(args)
    result = detection.predict(image)
    print(result.keys())
    # input_name = predictor.get_input_names()  # ['image', 'scale_factor']
    # input_image = predictor.get_input_handle('image')
    # input_image.copy_from_cpu(image)
    # input_scale_factor = predictor.get_input_handle('scale_factor')
    # input_scale_factor.copy_from_cpu(scale_factor)
    # predictor.run()
    # output_name = predictor.get_output_names()
    # output_tensor = predictor.get_output_handle(output_name[0])
    # pred = output_tensor.copy_to_cpu()
