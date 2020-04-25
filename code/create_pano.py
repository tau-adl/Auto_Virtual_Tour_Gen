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


#%% Input data
# A directory with N sub-directories each contains the images collected at one of the point in the route
# image_sequences_dir = r"C:\Users\shirang\Documents\myStuff\Project\panos\panos_office\image_sequence"
# image_sequences_dir = r"C:\Users\shirang\Documents\myStuff\Project\panos\panos_sinai\image_sequence_reordered_images_2"
# image_sequences_dir = r"C:\Users\shirang\Documents\myStuff\Project\panos\panos_sinai\image_sequence_2"
image_sequences_dir = "image_sequence"

suffix_in = '.jpg'
suffix_out = '.tif'

# krpano software directort
krpano_path = "C:\Users\shirang\Documents\myStuff\Project\krpano-1.20"

# Get number of points in route
n_points = len(glob.glob(os.path.join(image_sequences_dir, "*")))
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


# Create directory for all panoramas
panos_dir = os.path.join(image_sequences_dir, 'panoramas')
if not os.path.isdir(panos_dir):
    os.mkdir(panos_dir)


#%% Create panoramas from images using Hugin framework
bat_template_file = r"run_pano_from_dir_template.bat"

# create panorama for each point
for point in range(1, n_points+1):

    # get images path for the current point
    images_dir = os.path.join(image_sequences_dir, str(point))
    # get list of images of current point
    images_list = glob.glob(os.path.join(images_dir, '*' + suffix_in))
    images_string = " ".join(images_list)

    # create bat file for the current point
    bat_file = os.path.join(images_dir, 'run_pano_from_dir.bat')
    shutil.copyfile(bat_template_file, bat_file)

    # edit the bat file to run over the current point images
    for i, line in enumerate(fileinput.input(bat_file, inplace=True)):
        if (i == 0) :
            line = line.rstrip()
            sys.stdout.write(line + ' ' + images_dir + '\n')
        elif (i == 2) | (i == 14):
            line = line.rstrip()
            sys.stdout.write(line + ' ' + images_string + '\n')
        else:
            sys.stdout.write(line)

    # run bat file to create panorama for current point
    subprocess.run(bat_file, shell=True)

    # copy panorama file to the panoramas folder
    pano_path = os.path.join(images_dir, 'pano' + suffix_out)  # get name of panorama
    pano_path_to_copy = os.path.join(panos_dir, "pano" + str(point) + suffix_out)  # desired path to copy panorama
    if os.path.exists(pano_path):
        shutil.copyfile(pano_path, pano_path_to_copy)

print('')
print('done with panoramas')
print('')





#%% Create a virtual tour from several panoramas using krpano framework

# get list of panorama files
images_list = glob.glob(os.path.join(panos_dir, '*' + suffix_out))
images_string = " ".join(images_list)

# krpano bat file
# bat_file = r"C:\Users\shirang\Documents\myStuff\Project\krpano-1.20\make_pano.bat"
bat_file = os.path.join(krpano_path, 'make_pano.bat') # this is actually "MAKE PANO (NORMAL) droplet.bat" renames to have no spaces


# subprocess.run(bat_file + " " + images_string, shell=True)
sp = subprocess.Popen(bat_file + " " + images_string, stdin=subprocess.PIPE)
sp.stdin.write("\r\n") # send the CR/LF for pause
sp.stdin.close() # close so that it will proceed




#%% Insert hotspots by editing the vtour xml output file
# after xml is updated, the changes will be visible when opening the vtour from the exe file

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

points_left = [[-70,10],
               [-70,-10],
               [-90,-10],
               [-90,10]]

points_right = [[-10,10],
                [-10,-10],
                [10,-10],
                [10,10]]

point_dict = {'forward': points_forward, 'left': points_left, 'right': points_right}


# connect all panoramas using a single hotspot to create original direction route
for i in range(0, n_points-1):
    # get the appropriate hotspot location based on th route
    direction = route_directions[i]
    points = point_dict.get(direction)
    hotspot_node = create_hotspot("hs1", scenes_names[i+1], points)
    # scenes[i].append(hotspot_node)
    root.getchildren()[5+i].append(hotspot_node)
    # scenes[i].append(hotspot_node)


# print edited file
print(ET.tostring(root, encoding='utf8').decode('utf8')) # print full xml file

# write to file
xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent=" ")
with open(os.path.join(panos_dir, 'vtour', 'tour.xml'), "w") as f:
    f.write(xmlstr)




