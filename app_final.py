from flask import Flask, render_template, request, redirect, url_for, flash 
import cv2
import numpy as np
import mediapipe as mp
import pytesseract

app = Flask(__name__)

@app.route('/')
def form():
    return render_template('index.html')

@app.route('/', methods = ['POST']) #, 'GET'
def upload():
    if request.method == 'POST':
        f = request.files['file']
        print("vaibhav",f)    
        img1 = cv2.imread(f.filename)
        #rcParams['figure.figsize']=8,16

        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        #Image Processing

        def image_process1(img1):
            img_height = img1.shape[0]
            img_width = img1.shape[1] 
            face_detection_results = face_detection.process(img1[:,:,::-1])
            if face_detection_results.detections:
                for face_no, face in enumerate(face_detection_results.detections):
                    face_data = face.location_data

            #Declaring the variables for Calculating Bounding Box manually(for the new image i.e. rotated)

            rwidth  = face_data.relative_bounding_box.width
            rheight  = face_data.relative_bounding_box.height
            nor_width = rwidth*img_width
            nor_height = rheight*img_height
            box_width = 6.5*nor_width
            box_height = 10.4*nor_height
            nose_x = (face_data.relative_keypoints[2].x)*img_width
            nose_y = (face_data.relative_keypoints[2].y)*img_height
            ID_x = int((nose_x)-(box_width/2))
            ID_y = int((nose_y)-(box_height*0.45))

            #Declaring the Bounding Box(for the new image i.e. rotated)

            start_point = (ID_x, ID_y)
            end_point = (int(ID_x+box_width),int(ID_y+box_height))
            color = (255, 255, 255)
            thickness = 3
            img_copy = img1[:,:,::-1].copy()
            if face_detection_results.detections:
                for face_no, face in enumerate(face_detection_results.detections):
                    mp_drawing.draw_detection(image=img_copy, detection=face, keypoint_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),thickness=5,circle_radius=2))

            # Plotting the image with bounding box

            image1 = cv2.rectangle(img_copy, start_point, end_point, color, thickness)
            input_pts = np.float32([[ID_x,ID_y],[(ID_x+box_width),ID_y],
                                    [(ID_x+box_width),(ID_y+box_height)],[ID_x,(ID_y+box_height)]])
            output_pts = np.float32([[0,0],[img_width,0],[img_width,img_height],[0,img_height]])
            M = cv2.getPerspectiveTransform(input_pts,output_pts)
            out = cv2.warpPerspective(image1,M,(image1.shape[1],image1.shape[0]))
            return out
        
        a_out = image_process1(img1)

        #Image Contouring

        def contour1(img1,out):
            img_height = img1.shape[0]
            img_width = img1.shape[1] 
            face_detection_results = face_detection.process(img1[:,:,::-1])
            if face_detection_results.detections:
                for face_no, face in enumerate(face_detection_results.detections):
                    face_data = face.location_data

            #Declaring the variables for Calculating Bounding Box manually(for the new image i.e. rotated)

            rxmin  = face_data.relative_bounding_box.xmin
            rymin  = face_data.relative_bounding_box.ymin
            rwidth  = face_data.relative_bounding_box.width
            rheight  = face_data.relative_bounding_box.height
            nor_width = rwidth*img_width
            nor_height = rheight*img_height
            box_width = 6.5*nor_width
            box_height = 10.4*nor_height
            nose_x = (face_data.relative_keypoints[2].x)*img_width
            nose_y = (face_data.relative_keypoints[2].y)*img_height
            ID_x = int((nose_x)-(box_width/2))
            ID_y = int((nose_y)-(box_height*0.45))
            rex = (face_data.relative_keypoints[0].x)
            rey = (face_data.relative_keypoints[0].y)
            lex = (face_data.relative_keypoints[1].x)
            ley = (face_data.relative_keypoints[1].y)

            #convert img to grey
            img_grey = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
            #set a thresh
            thresh = 128
            #get threshold image
            ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
            thresh_img = cv2.erode(thresh_img , None , iterations=2)
            thresh_img = cv2.dilate(thresh_img , None , iterations=2)
            #find contours

            cnts, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            c = max(cnts, key=cv2.contourArea)

            # Obtain outer coordinates
            left = tuple(c[c[:, :, 0].argmin()][0])
            right = tuple(c[c[:, :, 0].argmax()][0])
            top = tuple(c[c[:, :, 1].argmin()][0])
            bottom = tuple(c[c[:, :, 1].argmax()][0])

            # Draw dots onto image
            cv2.drawContours(out, [c], -1, (36, 255, 12), 2)
            cv2.circle(out, left, 8, (0, 50, 255), -1)
            cv2.circle(out, right, 8, (0, 255, 255), -1)
            cv2.circle(out, top, 8, (255, 50, 0), -1)
            cv2.circle(out, bottom, 8, (255, 255, 0), -1)


            input_pts = np.float32([[ID_x,ID_y],[(ID_x+box_width),ID_y],[(ID_x+box_width),(ID_y+box_height)],[ID_x,(ID_y+box_height)]])
            output_pts = np.float32([[0,0],[3000,0],[right[0],4000],[0,4000]])
            M = cv2.getPerspectiveTransform(input_pts,output_pts)
            hi = cv2.warpPerspective(out,M,(out.shape[1],out.shape[0]))
            return hi

        #Image Data collection

        def get_data(out):
            string1 = pytesseract.image_to_string(out)
            def Convert(string):
                li = list(string.split(" "))
                return li
            return Convert(string1)   
        img_contour = contour1(img1,a_out)
        l1 = get_data(img_contour)
        if "Vellore" not in l1:
            l = get_data(a_out)
        else:
            l = get_data(img_contour)    
        print("list printing",l)
        return render_template('index.html',l=l)
#call main
if __name__ == "__main__":
    app.run(debug=True)

