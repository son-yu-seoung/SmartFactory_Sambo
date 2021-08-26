import sys
import math
import cv2 
import numpy as np

class LineDetection:

    def __init__(self):
        self.default_file = './image/green_7.24_1.jpg' # 나중에는 video에서 image로 받아야함.
    
    def hough_transform(self):
        lines_p = cv2.HoughLinesP(self.edge, 1, np.pi / 180, 50, None, 120 , 10)
        line_vertical = []
        line_horizontal = []

        if lines_p is not None:
            for i in range(0, len(lines_p)):
                l = lines_p[i][0] # 해당 직선을 지나는 두 점의 좌표 

                abs_v = abs(l[0] - l[2]) # x가 비슷하다면 수직 직선을 긋는다.
                abs_h = abs(l[1] - l[3]) # y가 비슷하다면 수평 직선을 긋는다.

                if abs_v <= 10 :  
                    l[1] = 0
                    l[3] = 500
            
                    line_vertical.append([[l[0], l[1]], [l[2], l[3]]]) # (n, 2, 2)
                    cv2.line(self.color_edge, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            
                elif abs_h <= 10:
                    l[0] = 0
                    l[2] = 500
                
                    line_horizontal.append([[l[0], l[1]], [l[2], l[3]]])
                    cv2.line(self.color_edge, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv2.LINE_AA)
        
        return line_vertical, line_horizontal
    
    def calculate_points(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        px = (x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)
        py= (x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)
        p = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)

        x = px / p
        y = py / p

        return x, y 


    def get_intersection_point(self, lines_v, lines_h):
        points = []

        for line_v in lines_v:
            for line_h in lines_h:
                x, y = self.calculate_points(line_v[0], line_v[1], line_h[0], line_h[1])
                x = round(x)
                y = round(y)

                points.append([x, y])
        
        return points

    def draw_circle(self, is_points):
        for x, y in is_points:
            cv2.circle(self.color_edge, (x, y), 5, (0, 255, 0), -1)


    def show_image(self):
        cv2.imshow("original", self.src)
        cv2.imshow("color_edge", self.color_edge)

        cv2.waitKey()
    
    def start_detection(self):

        # Loads an image
        self.src = cv2.imread(cv2.samples.findFile(self.default_file), cv2.IMREAD_GRAYSCALE)
        self.edge = cv2.Canny(self.src, 50, 200, None, 3 ) # Canny edge detector 
        self.color_edge = cv2.cvtColor(self.edge, cv2.COLOR_GRAY2BGR)

        line_vertical, line_horizontal = self.hough_transform()
        is_points = self.get_intersection_point(line_vertical, line_horizontal)

        self.draw_circle(is_points)
        self.show_image()

test = LineDetection()
test.start_detection()

        
         

