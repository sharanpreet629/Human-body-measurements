

import numpy as np
import sys
import os
import utils
import cv2

DATA_DIR = "data"



#  vertex = np.array(vertex).reshape(len(file_list), utils.V_NUM, 3)#utils.V_NUM
##  print("Vertex are of type = ",type(vertex)) 
##  print("vertex = ",vertex)
#  return vertex

def convert_cp():
    
  f = open(os.path.join(DATA_DIR, 'customBodyPoints.txt'), "r")

  tmplist = []
  cp = []
  for line in f:
    if '#' in line:
      if len(tmplist) != 0:
        cp.append(tmplist)
        tmplist = []
    elif len(line.split()) == 1:
      continue
    else:
      tmplist.append(list(map(float, line.strip().split())))
  cp.append(tmplist)


  return cp


# calculate measure data from given vertex by control points
def calc_measure(cp, vertex,height):
  measure_list = []
  
  for measure in cp:
    length = 0.0
    p2 = vertex[int(measure[0][1]), :]

    for i in range(0, len(measure)):
      p1 = p2
      if measure[i][0] == 1:
        p2 = vertex[int(measure[i][1]), :]  
        
      elif measure[i][0] == 2:
        p2 = vertex[int(measure[i][1]), :] * measure[i][3] + \
        vertex[int(measure[i][2]), :] * measure[i][4]
#        print("if 2 Measurement",int(measure[i][1]))
        
      else:
        p2 = vertex[int(measure[i][1]), :] * measure[i][4] + \
          vertex[int(measure[i][2]), :] * measure[i][5] + \
          vertex[int(measure[i][3]), :] * measure[i][6]
      length += np.sqrt(np.sum((p1 - p2)**2.0))

    measure_list.append(length * 100)# * 1000
  
  measure_list = float(height)*(measure_list/measure_list[0])
#  print("measure list = ",float(height)*(measure_list/measure_list[0])) 
  measure_list[8] = measure_list[8] * 0.36
  measure_list[3] = measure_list[3] * 0.6927
#  print("measure list = ",float(height)*(measure_list/measure_list[0]))
#  measure_list = float(height)*(measure_list/measure_list[0])
  return np.array(measure_list).reshape(utils.M_NUM, 1)



def extract_measurements(frame,img_path, height, vertices):
  genders = ["male"]#, "male"]
  measure = []
  for gender in genders:
    # generate and load control point from txt to npy file
    cp = convert_cp()

#    vertex = obj2npy(gender)[0]
    #calculte + convert
    measure = calc_measure(cp, vertices, height)

    for i in range(0, utils.M_NUM):
      num = measure[i][0]
      text = f"{utils.M_STR[i]}: {round(num,2)}"
      print(text)
      cv2.putText(frame, text , (10,(20+ i*40)), cv2.LINE_AA, 1, (0, 0, 0), 2)
      #cv2.imwrite("final_image.png",frame)
      #cv2.imshow("test", frame)
      # key = cv2.waitKey(1) & 0xFF
      # if key == ord('q'):
      #   print("VIDEO FEED TERMINATED")
      #   cv2.destroyAllWindows()

    face_path = './src/tf_smpl/smpl_faces.npy'
    faces = np.load(face_path)
    obj_mesh_name = 'test.obj'
    with open(obj_mesh_name, 'w') as fp:
        for v in vertices:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]))
        for f in faces:
            fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1))

        
    print("Model Saved..")


#if __name__ == "__main__":
#  extract_measurements()
  
#https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592 : to draw text on images and videos