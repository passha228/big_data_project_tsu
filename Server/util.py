def take_name_of_predict(number): 
    with open("./Names.txt",'r') as file: 
        for string in file: 
            if string.find(str(number)) != -1: 
                return string.split(", ")[-1] 

take_name_of_predict(2)