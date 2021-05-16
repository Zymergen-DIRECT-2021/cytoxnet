import csv
import os


def generate_column_name_list(csv_file_path):
    """
    Creates list of all column names in a CSV file.

    Parameters
    ----------
    csv_file_path: path to csv file to be formatted.

    Returns
    -------
    list_of_column_names: list of names of each column as 
    they appear in the input csv file.

    """
    with open(csv_file_path, 'r') as fp:
        # get list of column names
        csv_reader = csv.reader(fp, delimiter = ',')
  
        # list to store the names of columns
        list_of_column_names = []
  
        # loop to iterate through the rows of csv
        for row in csv_reader:
  
            # adding the first row
            list_of_column_names.append(row)
            
            list_of_column_names = list_of_column_names[0]
  
            # breaking the loop after the
            # first iteration itself
            break
    
    return list_of_column_names


def new_column_names(list_of_column_names):
    """
    Reformats column names to be more SQL friendly 
    (all lower-case, no spaces) and return new list.

    Parameters
    ----------
    list_of_column_names: list of the column names.

    Returns
    -------
    new_list_of_column_names: list of reformatted column names.

    """
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
    """
    Creates new CSV file based on existing CSV file with updated headers.

    Parameters
    ----------
    - new_list_of_column_names: list of column names which will be used as
    headers for the new csv file.
    - csv_file_path: file path of existing csv which the new csv will be
    based on.

    Returns
    -------
    saves a new csv file to same directory as input csv file path but
    with '_modified.csv' appended to the file name.

    """
    # create new file name based on name of input file
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
    """
    Wrapping function for reformatting the column names of a CSV
    file to be more 'SQL friendly' (all letters converted to lower-case
    and all spaces removed) and returning a modified CSV file within the
    same directory as the input CSV file.

    Parameters
    ----------
    csv_file_path: file path of existing csv which the new csv file will
    be based on.

    Returns
    -------
    saves a new csv file to same directory as input csv file path but
    with '_modified.csv' appended to the file name.

    """
    # retrieve list of headers from input csv file
    list_of_column_names = generate_column_name_list(csv_file_path)

    # modify the values within the list of headers
    new_list_of_column_names = new_column_names(list_of_column_names)

    # use modified headers to generate new csv file and save to same
    # directory as the input csv file.
    new_csv(new_list_of_column_names, csv_file_path)
    
    return
