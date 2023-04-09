# Dhruv Kamalesh Kumar
# 08-04-2023

# method to get the label
def getLabel(label):
    if label == 0:
        return "Heavy Plastic"
    elif label == 1:
        return "No Image"
    elif label == 2:
        return "No Plastic"
    elif label == 3:
        return "Some Plastic"
    else:
        return "Unknown"
