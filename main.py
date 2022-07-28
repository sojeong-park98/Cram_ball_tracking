import time
from dataloader import Image_loader, Videoloader
import cv2
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')
import torch
from torch.autograd import Variable
from utils import BaseTransform
from ssd import build_ssd

parser = argparse.ArgumentParser(description='HanQ')
parser.add_argument('--input', default='./input/2.mp4', type=str)
parser.add_argument('--data_type', default='video', type=str)
parser.add_argument('--save_vid', action='store_true')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

class Event_checker():
    def __init__(self, ball_path_list):
        self.ball_path_list = ball_path_list

    def tracker(self, show_img, ball_list):
        if len(ball_list)!=3:
            return show_img
        if 'red' in ball_list:
            self.ball_path_list['red'].append([int(ball_list['red']['x']), int(ball_list['red']['y'])])
            show_img = cv2.polylines(show_img, [np.array(self.ball_path_list['red'])], False, (0, 0, 1), 1)
        if 'yellow' in ball_list:
            self.ball_path_list['yellow'].append([int(ball_list['yellow']['x']), int(ball_list['yellow']['y'])])
            show_img = cv2.polylines(show_img, [np.array(self.ball_path_list['yellow'])], False, (0, 1, 1), 1)
        if 'white' in ball_list:
            self.ball_path_list['white'].append([int(ball_list['white']['x']), int(ball_list['white']['y'])])
            show_img = cv2.polylines(show_img, [np.array(self.ball_path_list['white'])], False, (1, 1, 1), 1)
        return show_img

def find_table(img):
    img = cv2.medianBlur(img, 5, 0)
    box_binary = 255-cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    box_binary = cv2.dilate(box_binary, None, iterations=3)

    left_list = []
    right_list = []
    top_list = []
    buttom_list = []
    for idx, w in enumerate(box_binary):
        if len(w[w==255]) > len(w[w==0]):
            if idx < box_binary.shape[0]*0.3:
                top_list.append(idx)
            elif idx > box_binary.shape[0]*0.7:
                buttom_list.append(idx)
    for idx, h in enumerate(box_binary.transpose(1,0)):
        if len(h[h==255]) > len(h[h==0]):
            if idx < box_binary.shape[1] * 0.3:
                left_list.append(idx)
            elif idx > box_binary.shape[1] * 0.7:
                right_list.append(idx)
    left_list = np.array(left_list)
    right_list = np.array(right_list)
    top_list = np.array(top_list)
    buttom_list = np.array(buttom_list)
    top = top_list.max()
    buttom = buttom_list.min()
    left = left_list.max()
    right = right_list.min()

    table = {'top': top,
             'buttom': buttom,
             'left': left,
             'right': right}

    return table


def find_ball(show_img, img_color):
    radius = None

    COLORS = [(1, 1, 1), (0, 0, 1), (0, 1, 1)]

    height, width = show_img.shape[:2]
    x = torch.from_numpy(transform(img_color)[0]).permute(2, 0, 1)
    if args.cuda:
        x = x.cuda()
    x = Variable(x.unsqueeze(0))
    with torch.no_grad():
        y = net(x)
    detections = y.data[0]
    scale = torch.Tensor([width, height, width, height])
    ball_list = {}

    for i in range(detections.size(0)):
        if detections[i, 0, 0] >= 0.3:

            pt = (detections[i, 0, 1:] * scale).cpu().numpy()
            cv2.rectangle(show_img,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          COLORS[i % 3], 2)
            pt = (detections[i, 0, 1:] * scale).cpu().numpy()
            x = (pt[0]+pt[2])/2
            y = (pt[1]+pt[3])/2
            radius_x = abs((int(pt[0])-int(pt[2]))/2)
            radius_y = abs((int(pt[1]) - int(pt[3])) / 2)
            radius = (radius_x+radius_y)/2

            if radius_x/radius_y>1.5 or radius_y/radius_x>1.5:
                continue

            cls=['','red', 'yellow', 'white']
            ball_info = {'color': cls[i],
                         'radius': int(radius),
                         'x':int(x),
                         'y':int(y)
                         }
            ball_list[cls[i]]=ball_info

    return show_img, ball_list, radius

if __name__ == '__main__':

    net = build_ssd('test', 300, 4, args.cuda)  # initialize SSD
    if args.cuda:
        net = net.cuda()
    net.load_state_dict(torch.load('CRAM.pth'))
    net.eval()
    transform = BaseTransform(net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))

    if args.save_vid:
        vid_writer = None
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()
            vid_path = 'output.mp4'
            vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1000, 500))
    if args.data_type=='video':
        imgs = Videoloader(args.input)
        fps = imgs.get_fps()
    else:
        imgs = Image_loader(args.input)

    start_time = time.time()
    ball_path_list = None
    sample = imgs[0]
    img, _, _ = sample
    table = find_table(img)
    ball_list = None
    event_checker = None

    WIDTH=1000
    HEIGHT=500

    for t, datas in enumerate(imgs):
        if datas==None:
            break
        img_original, img_color, name = datas
        img = img_original[table['top']:table['buttom'], table['left']:table['right']]
        img = cv2.resize(img, (WIDTH,HEIGHT), cv2.INTER_LINEAR)

        img_color = img_color[table['top']:table['buttom'], table['left']:table['right']]
        img_color = cv2.resize(img_color, (WIDTH, HEIGHT), cv2.INTER_LINEAR)

        show_img = np.zeros((img.shape[0], img.shape[1],3))
        show_img[:, :, 0] = img/255
        show_img[:, :, 1] = img/255
        show_img[:, :, 2] = img/255
        show_img = cv2.resize(show_img, (WIDTH, HEIGHT), cv2.INTER_LINEAR)

        if ball_path_list == None:
            show_img, ball_list, radius = find_ball(show_img, img_color)
            if len(ball_list) == 3:
                ball_path_list = {'red': [[ball_list['red']['x'], ball_list['red']['y']]],
                              'yellow': [[ball_list['yellow']['x'], ball_list['yellow']['y']]],
                              'white': [[ball_list['white']['x'], ball_list['white']['y']]]}
                event_checker = Event_checker(ball_path_list)
            else:
                continue
        else:
            show_img, ball_list, _ = find_ball(show_img, img_color)
            if ball_list != None:
                show_img = event_checker.tracker(show_img, ball_list)

        cv2.imshow("img", show_img)
        cv2.waitKey(1)
        if args.save_vid:
            vid_writer.write(np.uint8(show_img*255))

    cv2.destroyAllWindows()
    print("총 실행 시간: ", time.time()-start_time)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
