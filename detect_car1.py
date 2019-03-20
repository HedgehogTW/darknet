import darknet
# from pydarknet import Detector, Image
import cv2
import os
import argparse
import time
from time import localtime, strftime 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='darknet')   
    parser.add_argument('-f', action="store", dest='img_name', default='data/dog.jpg', help='input image file name')
    args = parser.parse_args()

    net = darknet.load_net('cfg/yolov3-spp.cfg'.encode("utf-8"), 'cfg/yolov3-spp.weights'.encode("utf-8"), 0)
    meta = darknet.load_meta("cfg/coco.data".encode("utf-8"))


    # net = Detector(bytes("cfg/densenet201.cfg", encoding="utf-8"), bytes("densenet201.weights", encoding="utf-8"), 0, bytes("cfg/imagenet1k.data",encoding="utf-8"))

    # net = Detector(bytes("cfg/yolov3-spp.cfg", encoding="utf-8"), bytes("cfg/yolov3-spp.weights", encoding="utf-8"), 0, bytes("cfg/coco.data",encoding="utf-8"))

    img_name = args.img_name.encode("utf-8")
    print('read img_name:', args.img_name)
    img = cv2.imread(args.img_name)

    # img2 = Image(img)
    # img2 = darknet.Image(img2)

    # r = net.classify(img2)
    start_time = time.time()
    # results = net.detect(img2,thresh=.2, hier_thresh=.2, nms=.2)
    results = darknet.detect_np(net, meta, img, thresh=.3, hier_thresh=.3, nms=.45)

    end_time = time.time()
               
    print('\tElapsed Time:{:.3f} sec'.format(end_time-start_time))

    
    min_car_height = 90
    frontcars=[]
    for i, obj in enumerate(results):
        cat, score, bounds = obj
        x, y, w, h = bounds

        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=1)
        cv2.putText(img,str(cat.decode("utf-8"))+str(i),(int(x),int(y)),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0))

        x, y, w, h = (int)(x), (int)(y), (int)(w), (int)(h)
        print('{:2d} {:10s} x{:4d}  y{:4d}  w{:4d}  h{:4d}  score {:.2f}'.format(i, str(cat.decode("utf-8")), x, y, w, h, score))

        if h >= min_car_height and (cat==b'car' or cat==b'truck'or cat==b'bus' or cat==b'motorbike'):
            frontcars.append(obj)



    dict_grid = {58:(100, 220, 140, 300), 57:(260, 370, 130,250), 56:(380, 465,110,210),
                        55:(466, 550,80,200)}
    notavailable=[]
    print('-------------------- frontcars')
    for i, obj in enumerate(frontcars):
        cat, score, bounds = obj
        x, y, w, h = bounds        
        x, y, w, h = (int)(x), (int)(y), (int)(w), (int)(h)

        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 255), thickness=2)
        print('{:2d} {:10s} x{:4d}  y{:4d}  w{:4d}  h{:4d}'.format(i, str(cat.decode("utf-8")), x, y, w, h))
        
        grid_bound = next(iter(dict_grid.values()))
        if len(grid_bound)==2:
            for gridID, (left,right) in dict_grid.items():
                if left < x < right:
                    notavailable.append(gridID)
                    # cv2.imwrite('car/{}_{:03d}.jpg'.format(cam_name, gridID), car)
                    break
        else:
            for gridID, (left,right, top, bottom) in dict_grid.items():
                if left < x < right and top < y < bottom:
                    notavailable.append(gridID)
                    # cv2.imwrite('car/{}_{:03d}.jpg'.format(cam_name, gridID), car)
                    break   
    notavailable.sort() 
    print('notavailable: ',notavailable)

    tmstr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    cv2.putText(img, tmstr, (5,img.shape[0]-20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0))


    cv2.imshow("output", img)
    # img2 = pydarknet.load_image(img)
    cv2.imwrite("output.jpg", img)
    cv2.waitKey(0)