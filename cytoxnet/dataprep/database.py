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

    # create list of sqlalchamy data types
    column_sqlalchemy_types = []
    for column_type in column_types:
        column_sqlalchemy_types.append(data_types.get(column_type))

    # create new dictionary with column names as keys and sqlalchemy data
    # types a values
    header_info = dict(zip(column_names, column_sqlalchemy_types))

    # connect to cytoxnet database
    engine = create_engine('sqlite:///cytoxnet.db', echo=True)
    engine.connect()
    meta = MetaData()

    # create SQL table from dataframe
    dataframe.to_sql(
        name=str(table_name),
        con=engine,
        if_exists='replace',
        index=False,
        dtype=header_info)


def tables_to_database(dataframe_dict):
    """
    Calls 'table_creater' function to generate multiple SQL tables at once from a dictionary
    with table names as the keys and dataframe objects as the values.  Such a dictionary is
    created using the cytoxnet.io.add_datasets function.

    Parameters
    ---------
    - dataframe_dict: dictionary with table names as keys and dataframe objects as values
    """
    for k, v in dataframe_dict.items():
        table_creator(k, v)


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
