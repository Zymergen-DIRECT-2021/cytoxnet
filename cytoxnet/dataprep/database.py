def drop_table(table_name):
    """
    Connects to cytoxnet.db database using sqlite and drops any
    table with the input name.

    Parameters
    ----------
    - table_name: str of name assigned to table to drop.
    """

    #Connecting to sqlite
    conn = sqlite3.connect('cytoxnet.db')

    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    #Doping EMPLOYEE table if already exists
    cursor.execute("DROP TABLE IF EXISTS "+table_name)

    #Commit your changes in the database
    conn.commit()

    #Closing the connection
    conn.close()


def table_creator(table_name, dataframe, codex=False, id_col=None):
    """
    Generates a SQL table based on an input dataframe object and saves it to
    the cytoxnet database.

    Parameters
    ----------
    - table_name: str of name assigned to table.
    - dataframe: dataframe object
    - codex: bool True if input dataframe is the compounds codex, otherwise False
    - id_col: int position of id column if exists
    """
    
    # reformat column names
    dataframe.columns= dataframe.columns.str.lower()
    dataframe.columns = dataframe.columns.str.replace(' ', '_')
    
    # insert 'ids' column at the first column position
    if codex == True:
        id_col_name_str = dataframe.columns[0]
        dataframe.rename(columns = {id_col_name_str:'ids'}, inplace = True)
    elif id_col != None:
        id_col_name_str = dataframe.columns[id_col]
        ids = dataframe.pop(id_col_name_str)
        dataframe.insert(0, 'ids', ids)
    else:
        dataframe.insert(0, 'ids', range(0, len(dataframe)))
    
    # shift column 'smiles' to second position
    second_column = dataframe.pop('smiles')
    
    # insert column using insert(position,column_name, first_column) function
    dataframe.insert(1, 'smiles', second_column)
    
    # dictionary to convert  between pandas dtypes and sqlalchemy dtypes
    data_types = {
        'object': String,
        'int64': Integer,
        'float64': Float,
        'bool': Boolean,
        'datetime64': DateTime,
        'timedelta[ns]': Interval,
        'category': String
    }

    # create list of column names and associated data types
    column_names = []
    column_types = []
    for column in dataframe.columns:
        column_names.append(column)
        column_types.append(dataframe[column].dtypes.name)


    # remove 'id' and 'smiles' columns from dataframe
    # this will be important when creating the database table
    del column_names[0:2]
    del column_types[0:2]
    
    # also remove 'foreign_key' column from dataframe
    if codex != True:
        del column_names[-1]
        del column_types[-1]
    else:
        pass

    # create list of sqlalchamy data types
    column_sqlalchemy_types = []
    for column_type in column_types:
        column_sqlalchemy_types.append(data_types.get(column_type))

    # create new dictionary with column names as keys and sqlalchemy data
    # types a values
    header_info = dict(zip(column_names, column_sqlalchemy_types))
    
    column_statements = []
    for column_name, column_type in header_info.items():
        column_statement = Column(column_name, column_type)
        column_statements.append(column_statement)

    # drop tables if they already exist within the database to prevent UNIQUE conflicts
    drop_table(table_name)
    
    # connect to cytoxnet database
    engine = create_engine('sqlite:///cytoxnet.db', echo=True)
    engine.connect()
    meta = MetaData()
    
    if codex != True:
        
        compounds = Table('compounds', meta, autoload=True, autoload_with=engine)
        
        new_table = Table(table_name, meta,
                          Column('ids', Integer, primary_key=True),
                          Column('smiles', String),
                          *column_statements,
                          Column('foreign_key', Integer, ForeignKey('compounds.ids')))
        
    else:
        
        codex_table = Table(table_name, meta,
                            Column('ids', Integer, primary_key=True),
                            Column('smiles', String),
                            *column_statements)
    
    # create table to sqlite 
    meta.create_all(engine)
    
    # create SQL table from dataframe
    dataframe.to_sql(
        name=str(table_name),
        con=engine,
        if_exists='append',
        index=False)
    
    return


def tables_to_database(dataframe_dict):
    """
    Calls 'table_creater' function to generate multiple SQL tables at once from a dictionary
    with table names as the keys and dataframe objects as the values.  Such a dictionary is
    created using the cytoxnet.io.add_datasets function.

    Parameters
    ---------
    - dataframe_dict: dictionary with table names as keys and dataframe objects as values.
    """
    for k, v in dataframe_dict.items():
        col_list = list(v.columns)
        if 'ids' in col_list:
            index = col_list.index('ids')
            table_creator(table_name=k, dataframe=v, codex=False, id_col=index)
        elif 'id' in col_list:
            index = col_list.index('id')
            table_creator(table_name=k, dataframe=v, codex=False, id_col=index)
        else:
            table_creator(table_name=k, dataframe=v, codex=False, id_col=None)
        
    return


def query_to_dataframe(tables, features_list=None):
    """
    Queryies selected tables within cytoxnet database, returning all columns from selected
    tables as well as the smiles
    """
    # connect to cytoxnet database
    engine = create_engine('sqlite:///cytoxnet.db', echo=False)
    meta = MetaData()

    # get list of tables to query
    if not isinstance(tables, list):
        tables = [tables]
    else:
        pass

    try:
        len(tables) <= 2
    except BaseException:
        print('Can only pass a maximum of two lists at a time.')

    table_object_list = []
    table_column_list = []

    # load existing tables
    compounds = Table('compounds', meta, autoload=True, autoload_with=engine)
    for table in tables:
        table = Table(str(table), meta, autoload=True, autoload_with=engine)

        # place table objects inside list
        table_object_list.append(table)

        # get list of column names and drop 'smiles' column
        column_list = []

        for c in table.c:
            if c.name != 'smiles':
                column_list.append(c.name)
            else:
                continue

        for column_name in column_list:
            table_column_list.append(
                getattr(
                    table.c,
                    column_name).label(
                    str(table) +
                    '_' +
                    column_name))

    # get list of features to include
    if features_list is not None:
        if not isinstance(features_list, list):
            features_list = [features_list]
        else:
            pass

        features_select = []

        for feature in features_list:
            features_select.append(getattr(compounds.c, feature))

        select_statement = [
            *table_column_list,
            compounds.c.smiles,
            *features_select]

    else:
        select_statement = [*table_column_list, compounds.c.smiles]

    # if more than one table being queried, perform a full-outer join
    if len(table_object_list) > 1:
        j_1 = table_object_list[0].join(
            table_object_list[1],
            table_object_list[0].c.smiles == table_object_list[1].c.smiles,
            isouter=True)
        j_2 = table_object_list[1].join(
            table_object_list[0],
            table_object_list[1].c.smiles == table_object_list[0].c.smiles,
            isouter=True)
        stmt_1 = select(select_statement).select_from(j_1).where(
            compounds.c.smiles == table_object_list[0].c.smiles).distinct()
        stmt_2 = select(select_statement).select_from(j_2).where(
            and_(
                compounds.c.smiles == table_object_list[1].c.smiles,
                table_object_list[0].c.smiles is None))
        stmt = union(*[stmt_1, stmt_2])

    elif len(table_object_list) == 1:
        stmt = select(select_statement).select_from(table_object_list[0].outerjoin(
            compounds, table_object_list[0].c.smiles == compounds.c.smiles))

    conn = engine.connect()
    dataframe = pd.read_sql(sql=stmt, con=conn)

    return dataframe