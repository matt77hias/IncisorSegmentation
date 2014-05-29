import cv2
import configuration as c

def whaha():
    cascade = cv2.CascadeClassifier("CV/data/Training/training/cascadesSCD1-u/cascade.xml")
    fname = c.get_fname_vis_pre(1, 'SCD')
    img = cv2.imread(fname)
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=(10, 10))
    #result is an array of x coordinate, y coordinate, weight, height for each rectangle
    rects[:,2:] += rects[:,:2]
    #result is an array of x coordinate, y coordinate, x + weight, y + height for each rectangle (opposite corners)
    
    for i in range(rects.shape[0]):
        cv2.rectangle(img, (rects[i,0], rects[i,1]), (rects[i,2], rects[i,3]), (0, 255, 0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    
if __name__ == '__main__':
    whaha()