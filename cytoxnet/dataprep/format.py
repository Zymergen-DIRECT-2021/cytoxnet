import csv
import os


def generate_column_name_list(csv_file_path):
    
    with open(csv_file_path, 'r') as fp:
        # get list of column names
        csv_reader = csv.reader(fp, delimiter = ',')
  
        # list to store the names of columns
        list_of_column_names = []
  
        # loop to iterate thorugh the rows of csv
        for row in csv_reader:
  
            # adding the first row
            list_of_column_names.append(row)
            
            list_of_column_names = list_of_column_names[0]
  
            # breaking the loop after the
            # first iteration itself
            break
    
    return list_of_column_names


def new_column_names(list_of_column_names):
    
    # list of new column headers
    new_list_of_column_names = []

    # convert column names to sql friendly strings
    for name in list_of_column_names:
        # make sure column name is string
        if type(name) != str:
            name = str(name)
        else:
            pass

        # make all names lowercase and replace spaces with underscores
        new_name = name.lower().replace(' ','_')

        # add new name to new list
        new_list_of_column_names.append(new_name)
    
    return new_list_of_column_names


def new_csv(new_list_of_column_names, csv_file_path):
    
    output_file = os.path.splitext(csv_file_path)[0] + "_modified.csv"
    
    with open(csv_file_path, 'r') as fp:
        # read the csv file using DictReader
        reader = csv.DictReader(fp, fieldnames=new_list_of_column_names)

        # use newline='' to avoid adding new CR at end of line
        with open(output_file, 'w', newline='') as fh: 
            writer = csv.DictWriter(fh, fieldnames=reader.fieldnames)
            writer.writeheader()
            header_mapping = next(reader)
            writer.writerows(reader)
    
    return


def format_csv(csv_file_path):
    list_of_column_names = generate_column_name_list(csv_file_path)
    new_list_of_column_names = new_column_names(list_of_column_names)
    new_csv(new_list_of_column_names, csv_file_path)
    
    return