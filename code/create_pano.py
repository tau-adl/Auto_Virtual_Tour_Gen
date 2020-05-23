#%% imports
import xml.etree.ElementTree as ET
import xml
from xml.dom import minidom
import copy
import os
import subprocess
import glob
import sys
import fileinput
import shutil


#%% Run configuration

# Setup

# Get\Set working directory
cwd = os.getcwd() # make sure you run in the code directory
cwd = '/home/yoni/Documents/Auto_Virtual_Tour_Gen/code'

# Set Hugin framework script
hugin_bash_template_file = os.path.join(cwd, 'run_pano_from_dir_template.sh')

# Set Krpano framework script
krpano_path = '/home/yoni/Documents/krpano-1.20.3'


# Input Data

# A directory with N sub-directories each contains the images collected at one of the point in the route
image_sequences_dir = os.path.join(cwd,'image_sequence')

suffix_in = '.[jJ][pP][gG]' #jpg suffix - either uppercase letters or lowercase
suffix_out = '.tif' # panoramas image format

# Get\Set number of points in route
n_points = len(glob.glob(os.path.join(image_sequences_dir, "*"))) # make sure only image folders are present
n_points = 9

# Direction of turns in every point in the route (should be at size n_points-1)
route_directions =   ['forward',
                      'forward',
                      'left',
                      'forward',
                      'left',
                      'forward',
                      'left',
                      'forward']


#%% Create panoramas from images using Hugin framework

# Create directory for all panoramas
panos_dir = os.path.join(image_sequences_dir, 'panoramas')
if not os.path.isdir(panos_dir):
    os.mkdir(panos_dir)
    
# create panorama for each point
for point in range(1, n_points+1):

    # get images path for the current point
    images_dir = os.path.join(image_sequences_dir, str(point))
    # get list of images of current point
    images_list = glob.glob(os.path.join(images_dir, '*' + suffix_in))
    images_string = " ".join(images_list)

    # create bat file for the current point
    bash_file = os.path.join(images_dir, 'run_pano_from_dir.sh')
    shutil.copyfile(hugin_bash_template_file, bash_file)
    shutil.copystat(hugin_bash_template_file, bash_file)


    # edit the bat file to run over the current point images
    for i, line in enumerate(fileinput.input(bash_file, inplace=True)):
        if (i == 0) :
            line = line.rstrip()
            sys.stdout.write(line + ' ' + images_dir + '\n')
        elif (i == 2) | (i == 14):
            line = line.rstrip()
            sys.stdout.write(line + ' ' + images_string + '\n')
        else:
            sys.stdout.write(line)

    # run bat file to create panorama for current point
    subprocess.run(bash_file, shell=True)

    # copy panorama file to the panoramas folder
    pano_path = os.path.join(images_dir, 'pano' + suffix_out)  # get name of panorama
    pano_path_to_copy = os.path.join(panos_dir, "pano" + str(point) + suffix_out)  # desired path to copy panorama
    if os.path.exists(pano_path):
        shutil.copyfile(pano_path, pano_path_to_copy)

print('')
print('Done with panoramas')
print('')


#%% Create a virtual tour from several panoramas using Krpano framework

# get list of panorama files
images_list = sorted(glob.glob(os.path.join(panos_dir, '*' + suffix_out)))
images_string = " ".join(images_list)

# krpano software directort
cmd = krpano_path + "/krpanotools makepano -config=templates/vtour-normal.config -panotype=cylinder"

vtour_dir = os.path.join(panos_dir, 'vtour')
if os.path.isdir(vtour_dir):
    os.rename(vtour_dir, vtour_dir+'_old')

p = subprocess.run(cmd + " " + images_string, shell=True)

print('')
print("Done with inital vtour. You can find it here: " + vtour_dir)
print('')


#%% Insert hotspots by editing the vtour xml output file
# after xml is updated, the changes will be visible when opening the vtour

xml_file = os.path.join(panos_dir, 'vtour', 'tour.xml')
tree = ET.parse(xml_file)
root = tree.getroot()

# print elements in file
print([elem.tag for elem in root.iter()])

# print full xml file
print(ET.tostring(root, encoding='utf8').decode('utf8'))

# print information for each element
for child in root:
    print(child.tag)
    print(child.attrib)

# search for elements of specific type
for scene in root.iter('scene'):
    print(scene.attrib)

# print scene list
for a in root.getchildren():
    print(a)
scenes = root.getchildren()[-n_points:]

# search for elements of specific type
for hotspot in root.iter('hotspot'):
    print(hotspot.attrib)

# print edited file
print(ET.tostring(root, encoding='utf8').decode('utf8')) # print full xml file

# create a style node and insert it
style_node = ET.Element("style")
style_node.set("name", "roomspot")
style_node.set("alpha", "0.6")
style_node.set("capture", "false")
root.insert(1, style_node)  # insert

# create an action node and insert it
action_node = ET.Element("action")
action_node.set("name", "goto")
action_node.text = "skin_loadscene(%1, OPENBLEND(0.8, 0.0, 0.6, 0.3, easeOutQuad));"
root.insert(2, action_node)  # insert

## create hotspot nodes to connect panoramas
# get scene names
scenes_names = [scenes[i].attrib['name'] for i in range(0, n_points)]

def create_hotspot(hotspot_name, target_scene_name, points):
    # inputs:
    # hotspot_name
    # target_scene_name - destination scene of the hotspot
    # p1, p2, p3, p4 - vertex points of the hotspot area, each in shape [x,y]

    hotspot_node = ET.Element("hotspot")
    hotspot_node.set("name", hotspot_name)
    hotspot_node.set("style", "roomspot|skin_tooltips")
    hotspot_node.set("tooltip", "click here")
    hotspot_node.set("onclick", 'goto({});'.format(target_scene_name))

    for p in points:
        point_node = ET.SubElement(hotspot_node, 'point')
        point_node.set("ath", str(p[0]))
        point_node.set("atv", str(p[1]))

    return hotspot_node

# hotspot locations for different type of turns in route
points_forward = [[-10,10],
                  [-10,-10],
                  [10,-10],
                  [10,10]]

points_left = [[-80,10],
               [-80,-10],
               [-100,-10],
               [-100,10]]

points_right = [[80,10],
               [80,-10],
               [100,-10],
               [100,10]]

points_back = [[170,10],
               [170,-10],
               [190,-10],
               [190,10]]

point_dict = {'forward': points_forward, 'left': points_left, 'right': points_right, 'back': points_back}

# scenes directions for the backeards tours
back_direction_of_scene_dict = {'forward': '180', 'left': '90', 'right': '-90', 'back': '0'}

# connect all panoramas using a single hotspot to create original direction route 
for i in range(0, n_points-1):
    
    # get the appropriate hotspot location based on th route
    direction = route_directions[i]
    points = point_dict.get(direction)
    hotspot_node = create_hotspot("hs1", scenes_names[i+1], points)
    # scenes[i].append(hotspot_node)
    root.getchildren()[5+i].append(hotspot_node)
    # scenes[i].append(hotspot_node)
    
    # add backward hotspot
    if i!=0:
        points = point_dict.get('back')
        hotspot_node = create_hotspot("hs2", "scene_back" + str(i), points)
        root.getchildren()[5+i].append(hotspot_node)

    # create additional copy of scene, for the way back with the proper direction
    # import copy
    scene_bck = copy.deepcopy(root.getchildren()[5+i]) 
    scene_bck.set("name", "scene_back" + str(i+1))
    back_scene_angle = back_direction_of_scene_dict[direction]
    scene_bck.getchildren()[0].set("hlookat", back_scene_angle)
    root.append(scene_bck)

# print edited file
print(ET.tostring(root, encoding='utf8').decode('utf8')) # print full xml file

# write to file
xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent=" ")
with open(os.path.join(panos_dir, 'vtour', 'tour.xml'), "w") as f:
    f.write(xmlstr)

print('')
print("Done adding hostspots to vtour. You can find it here: " + vtour_dir)
print('')


