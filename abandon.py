#####################################################################################################################
#     how to implement the alert launcher:                                                                          #
#                                                                                                                   #
# 1- create an instance of the object Alert launcher                                                                #
#     alertLauncher = Alert launcher(False)                                                                         #
#     If False, no output is performed (except alerts), if True, everything is printed (including alerts)           #
#                                                                                                                   #
# 2- launch the analysis of the outputs at each frame, by the following function, returns a frame color             #
#     couleurs = alertLauncher.analyse_outputs(outputs, image)                                                      #
#####################################################################################################################

import numpy as np
import cv2


'''
Helper Functions
'''
def meancolor(img,x_min,x_max,y_min,y_max):

    x_min,x_max,y_min,y_max = int(x_min),int(x_max),int(y_min),int(y_max)

    y =img.shape[0]
    x = img.shape[1]
    if (x_min <= 0):
        x_min = 0
    if (x_max >= x-1):
        x_max = x-1
    if (y_min <= 0):
        y_min = 0
    if (y_max >= y-1):
        y_max = y-1
    if (x_min>=x_max):
      x_max = x_min+1
    if (y_min>=y_max):
      y_min = y_max-1
    
    img_rect = img[y_min:y_max,x_min:x_max]
    try:
        img_rect = cv2.blur(img_rect,(20,20))
    except cv2.error:
        pass
    G,B,R = cv2.split(img_rect)
    meanG = np.mean(G)
    meanR = np.mean(R)
    meanB = np.mean(B)
    return (meanG,meanB,meanR)

def color_score(meanG1,meanR1,meanB1,meanG2,meanR2,meanB2):
    scoreG = np.abs(meanG1-meanG2)/255
    scoreR = np.abs(meanR1-meanR2)/255
    scoreB = np.abs(meanB1-meanB2)/255
    score = (scoreG+scoreB+scoreR)/3
    return score

def add_center(outputs): ## Add the center (x,y) of each bb object
  if not isinstance(outputs, list):
    result = np.zeros(shape=(outputs.shape[0], outputs.shape[1]+2))
    for index in range(outputs.shape[0]):
      result[index] = np.append(outputs[index], (outputs[index][1] - outputs[index][0], outputs[index][2] - outputs[index][3]))
    return result
  return outputs

  
def add_color(outputs, image): ## Get the mean RGB Color of each object
  if not isinstance(outputs, list):
    result = np.zeros(shape=(outputs.shape[0], outputs.shape[1]+3))
    for index in range(outputs.shape[0]):
      color_mean = meancolor(image, outputs[index][0], outputs[index][1], outputs[index][3], outputs[index][2])
      result[index] = np.append(outputs[index], (color_mean))
    return result
  return outputs


def prepare_output(track):
# transform track to numpy array, in the right format, also add center
  bbox = track.to_tlbr()
  class_name = track.get_class()
  if class_name == 'baggage' or class_name == 'handbag' or class_name == 'suitcase' or class_name == 'backpack':
    class_id = 0
  else: class_id = 1
  track_id = track.track_id
  output = np.array((int(bbox[0]),int(bbox[2]), int(bbox[1]), int(bbox[3]), track_id, class_id, int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])))
  return output


def dist_two_points( a_x, a_y, b_x, b_y):
    return np.sqrt( (a_y - b_y)**2 + (a_x - b_x)**2 )



class Entity:
  def __init__(self, pred = []):
    self.components = pred
    self.frame_immobile = 0
    self.abandon = False

  def get_components(self): return self.components

  def get_frame_immobile(self): return self.frame_immobile

class LanceurAlerte:
  def __init__(self, printer):
    self.listEntity = []
    self.printer = printer

  def get_listEntity(self):
    return self.listEntity

  def add_entity(self, entity):
    self.listEntity.append(entity)

  def add_pred(self, output):
    entity = Entity(output)
    self.add_entity(entity)

  def pred_in_list(self, output):
    object_id = int(output[5])
    if object_id == 1: # 0 => luggage,    1 => person,    we only want the luggage maybe float, to see if that poses a problem
      return -1
    elif object_id != 24 and object_id != 26 and object_id != 28 and object_id!= 63 and object_id != 64 :
      return -2
    else:
      for index, entity in enumerate(self.listEntity):
        distance_color = self.previous_frame_color_distance(output, index)
        if entity.components[4] == output[4] and entity.components[5] == object_id and distance_color < 0.7:
          return index # the prediction has the same id and the same class as an entity in the list
    return None # none of the entities in the list are the same as the predictions

  def privous_frame_distance(self, output, index):
    centre_a = (self.listEntity[index].components[6], self.listEntity[index].components[7])
    centre_b = (output[6], output[7])
    return dist_two_points(centre_a[0], centre_a[1], centre_b[0], centre_b[1])

  def previous_frame_color_distance(self, output, index):
    color_a = (self.listEntity[index].components[8], self.listEntity[index].components[9], self.listEntity[index].components[10])
    color_b = (output[8], output[9], output[10])
    return color_score(color_a[0], color_a[1], color_a[2], color_b[0], color_b[1], color_b[2])

  def previous_frame_person_distance(self,output,index):
    
    pass

  def analyse_outputs(self, outputs, image): #x1, x2, y1, y2, track_id, class_id
    list = []
    if self.printer: print("\n\nstart analysis of predictions")
    outputs = add_center(outputs) # index 6, 7
    outputs = add_color(outputs, image) # index 8, 9, 10
    for output in outputs:
      color = self.analyse_entity_v5(output)
      list.append(color)
    self.analyse_list()
    return list

  def analyse_entity_v5(self, output):
    index = self.pred_in_list(output)
    if index == -1:
      if self.printer: print("a prediction is a person, skip")
      return (255,35,0) ## Red
    if index == None:
      self.add_pred(output)
      if self.printer: print("adding prediction(luggage only) to entity_list")
      return (75,255,0) ## Green
    if index == -2:
      return 
    else:
      distance = self.privous_frame_distance(output, index)
      distance_color = self.previous_frame_color_distance(output, index)
      if self.printer: 
        print("distance = ", distance, "   distance_color = ", distance_color, "   updating entity")
      self.update_entity(output, index, distance, distance_color)
      if self.listEntity[index].abandon:
        return (0,45,255)
      else: 
        return (75,255,0)
      
  def update_entity(self, output, index, distance, distance_color):
    self.listEntity[index].components = output
    if self.printer:
      print("distance: ", distance)
      print("distance color: ", distance_color)
    if distance < 40 and distance_color < 0.2: ## Distance and color difference between current and previos frame
      self.listEntity[index].frame_immobile += 1 ## We want to add person distance
    else:
      self.listEntity[index].frame_immobile = 0

  def analyse_list(self):
    if self.printer: print("\nentity list analysis...")
    for entity in self.listEntity:
      if entity.frame_immobile >= 60:
        entity.abandon = True
        print("------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("/!\/!\/!\/!\/!\    ABANDON  ALERT  /!\/!\/!\/!\/!\ ")
        print("The entity number ", int(entity.components[4]), " with class number ", int(entity.components[5]), "  is considered ABANDONED! Check status")
        print("/!\/!\/!\/!\/!\    ABANDON  ALERT    /!\/!\/!\/!\/!\ ")
        print("------------------------------------------------------------------------------------------------------------------------------------------------------------")
      else:
        entity.abandon = False
    if self.printer: print("Done\n")
  
  def return_colors(self):
    liste = []
    not_abandon = (75,255,0)
    abandon = (255,35,0)
    for entity in self.listEntity:
      if entity.abandon:
        liste.append(abandon)
      else:
        liste.append(not_abandon)
    return liste
'''
  def analyse_entity_v4(self, track):
    output = prepare_output(track)
    index = self.pred_in_list(output)
    if index == -1:
      if self.printer: print("a prediction is a person, skip")
      return
    if index == -2:
      if self.printer: print("a prediction is not important")
      return 
    if index == None:
      self.add_pred(output)
      if self.printer: print("adding prediction(luggage only) to entity_list")
    else:
      distance = self.distance_frame_precedente(output, index)
      if self.printer: print("distance = ", distance, "  updating entity")
      self.update_entity(output, index, distance)   
'''