# pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用gradio将程序包装成一个可视化的界面，可以在网页可视化的展示
import gradio
import argparse
from predict import detection_class

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|在服务器上启动gradio服务|')
parser.add_argument('--model_path', default='model', type=str, help='|模型位置|')
parser.add_argument('--device', default='cpu', type=str, help='|设备|')
parser.add_argument('--threshold', default=0.8, type=float, help='|行人识别阈值，0.8为基准|')
parser.add_argument('--reid', default=True, type=bool, help='|是否对检测到的行人进行特征提取和匹配，返回人物编号0、1、2...|')
parser.add_argument('--reid_model_path', default='reid_model', type=str, help='|行人特征提取模型位置|')
parser.add_argument('--feature_database', default='feature_database.txt', type=str, help='|行人特征数据库|')
parser.add_argument('--feature_add', default=True, type=bool, help='|当行人特征数据库中没有此人时，是否增加其特征(暂时的)，False时不匹配返回None|')
parser.add_argument('--reid_threshold', default=0.85, type=float, help='|特征匹配阈值，0.85为基准|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def function(image):
    box_list, id_list = model.predict(image)
    image_draw = model.draw(image, box_list, id_list)
    return image_draw


if __name__ == '__main__':
    print('| 使用gradio启动服务 |')
    model = detection_class(args)
    gradio_app = gradio.Interface(fn=function, inputs=['image'], outputs=['image'])
    gradio_app.launch(share=False)
