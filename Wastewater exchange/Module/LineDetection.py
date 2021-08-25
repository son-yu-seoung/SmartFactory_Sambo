import sys
import math
import cv2 as cv
import numpy as np
def main(argv): # command창에서 LineDetection.py 호출시뒤에 문자열을 인자로 줄 수 있다. 
    
    default_file = '.image/green_7.24_1.jpg'
    filename = argv[0] if len(argv) > 0 else default_file # 만약 argv에 1개 이상의 문자가 들어있다면 argv[0](받은 인자)를 filename으로 지정한다.

    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    dst = cv.Canny(src, 50, 200, None, 3) # Canny edge detector 
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a))) # (x0, y0)
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a))) # (x1, y1)
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    
    
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            if l[0] == l[2]:
                l[1] = 0
                l[3] = 500
            elif l[1] == l[3]:
                l[0] = 0
                l[2] = 500
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Line", cdst)
    cv.imshow("Detected Lines (in red) - LineP", cdstP)
    
    cv.waitKey()
    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])