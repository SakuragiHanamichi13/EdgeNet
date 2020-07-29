import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import time



def visualization(rgb, model, device='cuda:0'):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)/255.0

    t_gray = torch.tensor(gray)
    t_gray = t_gray.view(1,1,gray.shape[0], gray.shape[1]).float()
    t_gray = t_gray.to(device)

    edge = model(t_gray)
    _,edgemap = torch.max(edge, 1)
    edgemap = edgemap.view(gray.shape[0], gray.shape[1])
    edgenp = (edgemap.to('cpu')).data.numpy()
    edgenp = (edgenp*255).astype(np.uint8)

    return edgenp


def demo(video_path, net, device='cuda:0'):
    save_w = 1280
    save_h = 360
    #
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('demo.avi', fourcc, 30, (save_w, save_h))

    cap = cv2.VideoCapture(video_path)
    
    while(1):
        ret, frame = cap.read()
        if ret==False:
            break
       
        frame = cv2.resize(frame,(640, 360),1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        #t1 = time.time()
        pred = visualization(frame, net)
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        #print('pred shape = ', pred.shape, 'frame shape = ', frame.shape)
        visual_frame = np.concatenate([frame, pred],axis = 1) 

        

        #t2 = time.time()
        #fps = 1.0/(t2-t1)
        #cv2.putText(visual_frame, 'fps:'+str(fps), (200, 120), 1,1,(0,0,255))
        
        out.write(visual_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.namedWindow('result', 2)
        cv2.imshow('result', visual_frame)
        cv2.waitKey(30)
    cap.release()
    cv2.destroyAllWindows()
