import xml.etree.ElementTree as ET


def get_Coordinates(file):
    # This part just read and return the coordinates part of the tooth and canal

    tree = ET.parse(file)
    root = tree.getroot()

    obj = root.findall('./outputs/object/item/polygon')
    Right_channel = []
    Right_tooth = []
    left_channel = []
    left_tooth = []
    points = []

    for name in root.iter('name'):
        points.append(name.text)

    for j in range(0, len(points)):
        for i in range(0, 100):
            if points[j] == 'Sağ M3':
                if (obj[j].find(("x" + str(i + 1)))) == None:
                    pass
                else:
                    Right_channel.append((int(float(obj[j].find("x" + str(i + 1)).text)),
                                          int(float(obj[j].find("y" + str(i + 1)).text))))
            elif points[j] == '48':
                if (obj[j].find(("x" + str(i + 1)))) == None:
                    pass
                else:
                    Right_tooth.append((int(float(obj[j].find("x" + str(i + 1)).text)),
                                        int(float(obj[j].find("y" + str(i + 1)).text))))
            elif points[j] == 'Sol M3':
                if (obj[j].find(("x" + str(i + 1)))) == None:
                    pass
                else:
                    left_channel.append((int(float(obj[j].find("x" + str(i + 1)).text)),
                                         int(float(obj[j].find("y" + str(i + 1)).text))))
            elif points[j] == '38':
                if (obj[j].find(("x" + str(i + 1)))) == None:
                    pass
                else:
                    left_tooth.append((int(float(obj[j].find("x" + str(i + 1)).text)),
                                       int(float(obj[j].find("y" + str(i + 1)).text))))

    return Right_channel, Right_tooth, left_channel, left_tooth
