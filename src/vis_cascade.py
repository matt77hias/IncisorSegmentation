import cv2
import configuration as c

def whaha():
    
    for m in ['l', 'u']:
        for i in c.get_trainingSamples_range():
            cascade = cv2.CascadeClassifier("CV/data/Training/training/cascadesSCD" + str(i) + '-' + m + "/cascade.xml")
            fname = c.get_fname_vis_pre(i, 'SCD')
            img = cv2.imread(fname)
            rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=(100, 100))
            #result is an array of x coordinate, y coordinate, weight, height for each rectangle
            rects[:,2:] += rects[:,:2]
            #result is an array of x coordinate, y coordinate, x + weight, y + height for each rectangle (opposite corners)
        
            for r in range(rects.shape[0]):
                cv2.rectangle(img, (rects[r,0], rects[r,1]), (rects[r,2], rects[r,3]), (0, 255, 0), 2)
            fname = 'test' + str(i) + '-' + m + '.png'
            cv2.imwrite(fname, img)
    
if __name__ == '__main__':
    whaha()