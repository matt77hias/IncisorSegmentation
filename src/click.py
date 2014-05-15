import cv2
import numpy as np
import configuration as c

img = None
count = 0
landmarks = np.zeros((c.get_nb_dim()))
    
def draw_landmark(event, x, y, flags, param):
    if (event == cv2.EVENT_LBUTTONDOWN and count < c.get_nb_landmarks()):
        global img, count, landmarks
        landmarks[(2*count)] = float(x)
        landmarks[(2*count+1)] = float(y)
        if count > 0:
            xp = int(landmarks[(2*count-2)])
            yp = int(landmarks[(2*count-1)])
            cv2.line(img, (xp,yp), (x,y), (0,0,255))
        cv2.circle(img,(x,y),2,(0,0,255), -1)  
        count += 1  
        if count == c.get_nb_landmarks():
            xp = int(landmarks[0])
            yp = int(landmarks[1])
            cv2.line(img, (x,y), (xp,yp), (0,0,255))

def init_params(fname):
    global img, count, landmarks
    count = 0
    landmarks = np.zeros((c.get_nb_dim()))
    img = cv2.imread(fname)
                       
def draw_landmarks(fname):
    init_params(fname)
    img_name = 'Click left to add point, click esc to terminate'
    cv2.namedWindow(img_name)
    cv2.setMouseCallback(img_name, draw_landmark)

    while(1):
        cv2.imshow(img_name, img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    return landmarks
    
def draw_and_write(nr_trainingSample, nr_tooth, method=''):
    selected = draw_landmarks(c.get_fname_vis_pre(nr_trainingSample, method))    
    fname = c.get_fname_fitting_manual_landmark(nr_trainingSample, nr_tooth)   
    selected.tofile(fname, sep=" ", format="%s")

if __name__ == "__main__":
    draw_and_write(nr_trainingSample=1, nr_tooth=1, 'SCD')