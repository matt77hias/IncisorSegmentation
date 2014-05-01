import cv2, cv
import configuration as c

def on_mouse(event, x, y, flags):
    
    print ('ok')
    
    if event == cv.CV_EVENT_LBUTTONDOWN:
        print 'Start Mouse Position: '+str(x)+', '+str(y)
    elif event == cv.CV_EVENT_LBUTTONUP:
        print 'End Mouse Position: '+str(x)+', '+str(y)
       
        
        
if __name__ == "__main__":
    img = cv2.imread(c.get_fname_vis_pre(1, 'SCD'))
    img_name = 'Click left to add point, click right to terminate'
    cv.SetMouseCallback(img_name, on_mouse, 0)
    
    cv2.namedWindow(img_name)
    cv2.imshow(img_name, img)