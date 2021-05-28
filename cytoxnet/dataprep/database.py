def table_creator(table_name, dataframe):
    """
    Generates a SQL table based on an input dataframe object and saves it to
    the cytoxnet database.

    Parameters
    ----------
    - table_name: str of name assigned to table.
    - dataframe: dataframe object
    """
    
    # dictionary to convert  between pandas dtypes and sqlalchemy dtypes    
    data_types = {
        'object': String,
        'int64': Integer,
        'float64': Float,
        'bool': Boolean,
        'datetime64':DateTime,
        'timedelta[ns]':Interval,
        'category':String
    }
    
    # create list of column names and associated data types
    column_names = []
    column_types = []
    for column in dataframe.columns:
        column_names.append(column)
        column_types.append(dataframe[column].dtypes.name)
    
    print(column_types)
        
    # remove 'id' and 'smiles' columns from dataframe
    # this will be important when creating the database table
    del column_names[0:2]
    del column_types[0:2]
    
    # create list of sqlalchamy data types
    column_sqlalchemy_types = []
    for column_type in column_types:
        column_sqlalchemy_types.append(data_types.get(column_type))
        
    # create new dictionary with column names as keys and sqlalchemy data types a values
    header_info = dict(zip(column_names, column_sqlalchemy_types))
    #print(header_info)
    
    # connect to cytoxnet database
    engine = create_engine('sqlite:///cytoxnet.db', echo = True)
    engine.connect()
    meta = MetaData()
    
    # create SQL table from dataframe
    dataframe.to_sql(name=str(table_name), con=engine, if_exists='replace', index=False, dtype=header_info)



#def tables_to_database(dataframe_dict):
    """
    Calls 'table_creater' function to generate multiple SQL tables at once from a dictionary
    with table names as the keys and dataframe objects as the values.  Such a dictionary is
    created using the cytoxnet.io.add_datasets function.

    Parameters
    ---------
    - dataframe_dict: dictionary with table names as keys and dataframe objects as values
    """

    # call 'table_creater' function and iterate through dictionary



def query_to_dataframe(tables, features_list=None):
    """
    Queryies selected tables within cytoxnet database, returning all columns from selected
    tables as well as the smiles 
    """
    # connect to cytoxnet database
    engine = create_engine('sqlite:///cytoxnet.db', echo = False)
    meta = MetaData()

    # get list of tables to query
    if type(tables) != list:
        tables = [tables]
    else:
        pass
    
    tables_to_query = []
    table_columns = []
    
    # load existing tables
    compounds = Table('compounds', meta, autoload=True, autoload_with=engine)
    for table in tables:
        table = Table(str(table), meta, autoload=True, autoload_with=engine)
        
        # get list of column names and drop 'smiles' column
        column_list = []
        
        for c in table.c:
            if c.name !='smiles':
                column_list.append(c.name)
            else:
                continue
            
        for column_name in column_list:
            table_columns.append(getattr(table.c, column_name).label(str(table)+'_'+column_name))
            
        # create statements to include in query for each table
        tables_to_query.append(table.c.smiles == compounds.c.smiles)
        
    # get list of features to include

    if features_list != None:    
    
        if type(features_list) != list:
            features_list = [features_list]
        else:
            pass
    
        select_stm = []
    
        for feature in features_list:
            select_stm.append(getattr(compounds.c, feature))
        
        # generate select statement
        stmt = select([*table_columns, compounds.c.smiles, *select_stm]).where(or_(*tables_to_query)).distinct()
                        
    else:
        # generate select statment
        stmt = select(*table_columns, compounds.c.smiles).where(or_(*tables_to_query)).distinct()
    
    conn = engine.connect()
    dataframe = pd.read_sql(sql=stmt, con=conn)

    return dataframe