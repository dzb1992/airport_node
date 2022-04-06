# WS client example
#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time
from event import *
import cv2
import numpy as np
from numpy import mean
from PIL import Image
from utils_node import *
from yolo import YOLO
import asyncio
import websockets
yolo = YOLO()


async def node_detect():
    # uri = "ws://10.21.6.14:8765"
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # ----------------------------------------------------------------------------------------------------------#
        mode = "video"
        # ----------------------------------------------------------------------------------------------------------#
        #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
        #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
        #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
        #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
        #   video_fps用于保存的视频的fps
        #   video_path、video_save_path和video_fps仅在mode='video'时有效
        #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
        # ----------------------------------------------------------------------------------------------------------#
        video_path = "./img/input_video.mp4"
        video_save_path = "res.mp4"
        video_fps = 25.0
        # -------------------------------------------------------------------------#
        #   test_interval用于指定测量fps的时候，图片检测的次数
        #   理论上test_interval越大，fps越准确。
        # -------------------------------------------------------------------------#
        test_interval = 100
        # -------------------------------------------------------------------------#
        #   dir_origin_path指定了用于检测的图片的文件夹路径
        #   dir_save_path指定了检测完图片的保存路径
        #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
        # -------------------------------------------------------------------------#
        dir_origin_path = "img/"
        dir_save_path = "img_out/"
        bottom_dist = [0, 0, 0, 0, 0]
        # -------------------------------------------------------------------------#
        #   nodes 的定义列表
        # -------------------------------------------------------------------------#
        nodes = ["aeroplane out",
                 "aeroplane in place",
                 "dining_car out",
                 "catering",
                 "refueling_truck out",
                 "refueling", ]
        # -------------------------------------------------------------------------#
        #   可视化区域属性自定义
        #   ptCenter中心点位置
        #   axesSize长轴半径为 90，短轴半径为 60
        #   旋转角度为
        # -------------------------------------------------------------------------#
        font = cv2.FONT_ITALIC
        ptCenter = (896, 166)
        axesSize = (365, 86)
        rotateAngle = 0
        startAngle = 0
        endAngle = 360
        # -------------------------------------------------------------------------#
        #   保障节点展示位置
        # -------------------------------------------------------------------------#
        playnodes = [(16, 200), (16, 230), (16, 260)]
        # -------------------------------------------------------------------------#
        #   画保障节点前边的圆圈
        # -------------------------------------------------------------------------#
        point_color = (0, 255, 0)
        thickness = 1
        lineType = 4
        font = cv2.FONT_ITALIC
        coord_play = [(4, 194), (4, 224), (4, 254)]
        # -------------------------------------------------------------------------#
        #  init 初始化
        # -------------------------------------------------------------------------#
        bottom_dist = [0, 0, 0, 0, 0]
        center_l = [0, 0, 0, 0, 0]
        center_ref_truck = [0, 0, 0, 0, 0]
        # ---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        # ---------------------------------------------------------#
        npz_data = np.load('./model_data/grid_date_26985.npz', allow_pickle=True)
        coord, id = npz_data['coord'], npz_data['id']
        # # 根据坐标，找出对应区域
        id_Position = [tuple(k) for k in npz_data['id']]
        model_msg = model_ready_event()
        await websocket.send(model_msg)

        if mode == "predict":
            '''
            1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
            2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
            3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
            在原图上利用矩阵的方式进行截取。
            4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
            比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
            '''
            while True:
                img = input('Input image filename:')
                try:
                    image = Image.open(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    r_image = yolo.detect_image(image)
                    r_image.show()

        elif mode == "video":
            capture = cv2.VideoCapture(video_path)
            total_fps = capture.get(7)
            if video_save_path != "":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

            ref, frame = capture.read()
            if not ref:
                raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

            fps = 0.0
            count = 0
            while (True):
                t1 = time.time()
                # 读取某一帧
                ref, frame = capture.read()
                if not ref:
                    break
                count += 1
                if count % 600 == 0:
                    state_flow = [0, 0, 0]
                    # 格式转变，BGRtoRGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 转变成Image
                    frame = Image.fromarray(np.uint8(frame))
                    # 进行检测
                    frame1, out_boxes, out_classes = yolo.detect_image(frame)
                    frame = np.array(frame1)
                    # node logic
                    label_set = set(out_classes.tolist())
                    class_names = ["aeroplane", "dining_car", "refueling_truck"]
                    for i, c in list(enumerate(out_classes)):
                        predicted_class = class_names[int(c)]
                        box = out_boxes[i]
                        top, left, bottom, right = box

                        id_dist = []
                        center_Position = (int(left + (right - left) / 2), int(top + (bottom - top) / 2))
                        for v in id_Position:
                            x = np.array(list(v))
                            box_find = np.array(center_Position)
                            dist = np.sqrt(np.sum(np.square(x - box_find)))
                            id_dist.append(dist)
                        index = np.array(id_dist).argmin()
                        frame = cv2.putText(frame, "class: %s id: %s " % (predicted_class, (index)), (0, (i + 2) * 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                    if 0 in out_classes.tolist():
                        for i, c in list(enumerate(out_classes)):
                            # 检测飞机的入位情况
                            if c == 0:
                                bottom_center = [int(out_boxes[i][1] + (out_boxes[i][3] - out_boxes[i][1]) / 2),
                                                 int(out_boxes[i][2])]
                                bool_aeroplan = (isPoiWithinBox(bottom_center, [[810, 656], [966, 981]], toler=0.0001))
                                # bool_aeroplan_out = (isPoiWithinBox(bottom_center, [[246, 254], [1550, 706]], toler=0.0001))
                                # end_dis = calculate_distance([887, 975], bottom_center)
                                # bottom_dist.append(int(out_boxes[i][2]))
                                # bottom_dist.pop(0)
                                # delta = out_boxes[i][2] - mean(bottom_dist)
                                # # print("delta%s"%delta)

                                # 飞机就位logic
                                if bool_aeroplan == False:
                                    frame = cv2.putText(frame, nodes[0], playnodes[0], font, 1, (0, 0, 255), thickness=2, )
                                    print(nodes[0])
                                    state_flow[0] = 0
                                elif bool_aeroplan == True:
                                    frame = cv2.putText(frame, nodes[1], playnodes[0], font, 1, (0, 0, 255), thickness=2, )
                                    print(nodes[1])
                                    state_flow[0] = 1

                            # 餐车的计算
                            elif int(c) == 1:
                                center = (int(out_boxes[i][1] + (out_boxes[i][3] - out_boxes[i][1]) / 2),
                                          int(out_boxes[i][0] + (out_boxes[i][2] - out_boxes[i][0]) / 2))
                                end_dis_dining = calculate_distance([1140, 323], center)
                                center_l.append(end_dis_dining)
                                center_l.pop(0)
                                delta_center = abs(end_dis_dining - mean(center_l))
                                center_l_c = (int(out_boxes[i][1]),
                                              int(out_boxes[i][0] + (out_boxes[i][2] - out_boxes[i][0]) / 2))
                                bool_dining = (isPoiWithinBox(list(center_l_c), [[966, 255], [1024, 399]], toler=0.0001))
                                bool_dining_out = (
                                    isPoiWithinBox(list(center_l_c), [[246, 254], [1550, 706]], toler=0.0001))

                                # 餐车配餐逻辑
                                if (bool_dining == False) or (bool_dining == False and bool_dining_out == False):
                                    frame = cv2.putText(frame, nodes[2], playnodes[1], font, 1, (0, 0, 255), thickness=2, )
                                    state_flow[1] = 0
                                elif bool_dining == True:
                                    frame = cv2.putText(frame, nodes[3], playnodes[1], font, 1, (0, 255, 0), thickness=2, )
                                    state_flow[1] = 1

                            # 对燃油车车进行检测判断
                            elif int(c) == 2:
                                center_truck = (int(out_boxes[i][1] + (out_boxes[i][3] - out_boxes[i][1]) / 2),
                                                int(out_boxes[i][0] + (out_boxes[i][2] - out_boxes[i][0]) / 2))
                                end_dis_ref_truck = calculate_distance([485, 502], center_truck)
                                center_ref_truck.append(end_dis_ref_truck)
                                center_ref_truck.pop(0)
                                bool_ref_truck = (
                                    isPoiWithinBox([center_truck[0], center_truck[1]], [[395, 442], [568, 564]],
                                                   toler=0.0001))
                                bool_ref_truck_out = (
                                    isPoiWithinBox([center_truck[0], center_truck[1]], [[246, 254], [1550, 706]],
                                                   toler=0.0001))
                                delta_center_ref_truck = abs(end_dis_ref_truck - mean(end_dis_ref_truck))
                                if bool_ref_truck == False or (bool_ref_truck == False and bool_ref_truck_out == True):
                                    frame = cv2.putText(frame, nodes[4], playnodes[2], font, 1, (0, 0, 255), thickness=2, )
                                    state_flow[2] = 0
                                elif bool_ref_truck == True:
                                    frame = cv2.putText(frame, nodes[5], playnodes[2], font, 1, (0, 255, 0), thickness=2, )
                                    state_flow[2] = 1

                        print(state_flow)
                        name = str(state_flow[0]) + "," + str(state_flow[1]) + "," + str(state_flow[2])
                        await websocket.send(name)
                        #print(f"> {name}")
                        print("发送给服务端的信息：%s"%name)
                        greeting = await websocket.recv()
                        #print(f"< {greeting}")
                    else:
                        state_flow = [0, 0, 0]
                        print("There are no aeroplane on the apron ")
                        name = str(state_flow[0]) + "," + str(state_flow[1]) + "," + str(state_flow[2])
                        await websocket.send(name)
                        #print(f"> {name}")

                        greeting = await websocket.recv()
                        #print(f"< {greeting}")

                    # 在视频上画网格和ID
                    for i in range(coord.shape[0]):
                        cv2.line(frame, tuple(coord[i][0]), tuple(coord[i][1]), point_color, thickness, lineType)
                    id_data = dict()
                    for id_num, id_coor in enumerate(id):
                        id_data[id_num] = tuple(id_coor)
                        frame = cv2.putText(frame, str(id_num), tuple(id_coor), font, 0.5, (0, 0, 0), thickness=1, )

                    # 在节点前画圆圈
                    for center_p in coord_play:
                        cv2.circle(frame, center_p, 8, (0, 0, 255), thickness, lineType)
                    # RGBtoBGR满足opencv显示格式
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.ellipse(frame, ptCenter, axesSize, rotateAngle, startAngle, endAngle, point_color, thickness,
                                lineType)
                    fps = (fps + (1. / (time.time() - t1))) / 2
                    print("fps= %.2f" % (fps))
                    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    #cv2.imshow("video", frame)
                    c = cv2.waitKey(1) & 0xff
                    if video_save_path != "":
                        out.write(frame)

                    if c == 27:
                        capture.release()
                        break

            print("Video Detection Done!")
            capture.release()
            if video_save_path != "":
                model_msg = model_end_event()
                await websocket.send(model_msg)
                print("Save processed video to the path :" + video_save_path)
                out.release()
            cv2.destroyAllWindows()

        elif mode == "fps":
            img = Image.open('img/street.jpg')
            tact_time = yolo.get_FPS(img, test_interval)
            print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

        elif mode == "dir_predict":
            import os

            from tqdm import tqdm

            img_names = os.listdir(dir_origin_path)
            for img_name in tqdm(img_names):
                if img_name.lower().endswith(
                        ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    image_path = os.path.join(dir_origin_path, img_name)
                    image = Image.open(image_path)
                    r_image = yolo.detect_image(image)
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    r_image.save(os.path.join(dir_save_path, img_name))

        else:
            raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")


asyncio.get_event_loop().run_until_complete(node_detect())


