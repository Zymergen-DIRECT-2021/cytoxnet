import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
import os
# Third party imports 
import time
import collections
import json
import pandas as pd
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pickle

import requests as rq

pd.options.mode.chained_assignment = None 

logging.basicConfig(level=logging.WARNING)

requests = rq.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
requests.mount('http://', HTTPAdapter(max_retries=retries))

url = "http://classyfire.wishartlab.com"
proxy_url =  "https://gnps-classyfire.ucsd.edu"



def structure_query(compound, label='pyclassyfire'):
    """
    Submit a or many compounds with string representation (SMILES or INChE) to ClassyFire api for classification and recieve the corresponding query id.
    
    Arguments:
        compound - string of compounds. for more than one compound, seperate by '\\n'
            type == str
            eg. for two compounds to be submitted with SMILES string: 'OCC(CC#C)NC=O\\nC[C@@H](O)C#CCC(=O)C'
            
        label - string associated with the label for this query. Does not have to be unique
            type == str
            D 'pyclassyfire'
    """
    assert type(compound) == str, 'The incoming compound/s are not in string format'
    
    ## post the compounds and recieve the response, if timeout, try again 5 times
    error_counter=0
    while True:
        try:
            r = requests.post(url + '/queries.json', timeout=20, data='{"label": "%s", '
                              '"query_input": "%s", "query_type": "STRUCTURE"}'
                                                          % (label, compound),
                              headers={"Content-Type": "application/json", 'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'})
            
            ## success, reset counter
            error_counter=0
            
            break
        except (rq.ReadTimeout, rq.ConnectionError):
            ## add to counter, don't want to run forever
            error_counter +=1
            print('response timeout or connection error, trying again')
            if error_counter == 5:
                print('Too many errors, aborting')
                raise
        
    
    ## check response status
    r.raise_for_status()
    
    ## return the query id
    return r.json()['id']


def get_results(query_id, return_format="json"):
    """
    Retrieve the results of a query from ClassyFire API. Currently API only accepts json return format
    
    Arguments:
        query_id - integer id of the query to retrieve results from.
            type == int 
            eg. 5672358
            
        return_format - format for response to follow. 
            type == str, valid format eg. 'json', 'csv'
            D 'json'
            
    Returns:
        results - results in the requested format from the GET response.
            type == str
    """
    assert type(return_format) == str, 'The return format argument should be a string associate with the desired format type, eg. "json"'
    
    ## GET request and return response
    while True:
        try:
            r = requests.get('%s/queries/%s.%s' % (url, query_id, return_format), timeout=30,
                             headers={"Content-Type": "application/%s" % return_format, 
                                      'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'})

            ## check the status of response
            r.raise_for_status()

            ## return the contents of the response
            results = r.text
            return results
        except rq.ReadTimeout:
            print('response timeout, trying again')


                
                
def dataframe_query(df, structure_key, chunk_size = 10, query_dict=None, **kwargs):
    """
    Submit queries of compound strings to ClassyFire api from a column in a pandas dataframe. 
    
    Arguments:
        df - pandas dataframe that contains compounds to be classified
            type == pd.DataFrame
            
        structure_key - column name in df that contains compound structures to submit
            type == str
            
        chunk_size - number of compounds to submit per query. API currently only supports 10 maximum.
            type == int
            D 10
            
        query_dict - If the user wishes to pick up from a failed query, start from the dumped results of the last attempt. Dictionary of the same as output.
            type == Dict
            D None
    
    Returns:
        query_dict - dictionary with query ids submitted as keys, and list of compounds submited as values
            type == dict
            eg. if two queries of two compounds each were submited:
                {3526547: ['<SMILES1>', '<SMILES2>'],
                 3526548: ['<SMILES3>', '<SMILES4>']}
    
    """
    assert type(df) == pd.core.frame.DataFrame, 'incoming df is not a dataframe'
    assert type(structure_key) == str, 'structure_key should be a string'
    assert type(chunk_size) == int, 'chunk_size must be an int (recommended <10)'
    assert structure_key in df.columns, 'structure_key was not found in the incoming dataframe'
    
    ## if the user is picking up where left off
    if query_dict:
        assert type(query_dict == dict), 'query_dict must be a dictionary with query ids as keys and species lists as values'
        
        ## remove from df the already queried molecules
        df_copy = df.copy()
        
        #. already queried species
        species = []
        for l in query_dict.values():
            species.extend(l)
            
        print('Loaded dict length: ', len(query_dict))  
        print('Loaded species length: ', len(species))
        print('Loaded DF length: ', len(df_copy))
            
        #. remove from input dataframe
        df_copy = df_copy.iloc[len(species):]
        print('Rectified DF length: ', len(df_copy))
        
        #. if its empty, as in if the whole dataframe has been queried
        if df_copy.empty:
            print('All species have already been queried, returning input query dictionary')
            
            return query_dict
        
    else:
        ## initialize objects
        query_dict = {}
        df_copy = df.copy()

    comps = [] 
    
    try:
        ## loop through all compounds in column
        for s in list(df_copy[structure_key]):

            #. add them to a list
            comps.append(s)

            #. if list hits chunk size, submit query, record in query dict, and continue
            if not len(comps) % chunk_size:
                query_id = structure_query('\\n'.join(comps))
                query_dict[query_id] = comps
                comps = []
                print('submitted {}'.format(str(query_id)))
                time.sleep(1)

        #. if list never hits chunk size, just submit and record them all
        if comps:
            query_id = structure_query('\\n'.join(comps))
            print(query_id)
            query_dict[query_id] = comps

        ## report
        print('%s queries submitted to ClassyFire API' % len(query_dict))
        print('IDs: {}'.format(query_dict.keys()))
        
    except:
        ## save data if error
        print('Dumping query dict to "qd.pkl"')
        with open('qd.pkl', 'wb') as handle:
            pickle.dump(query_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        raise
    
    ## pickle all of the queries
    with open('qd.pkl', 'wb') as handle:
        pickle.dump(query_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
    return query_dict

def query_dict_return_patient(query_dict, 
                             max_wait=100, wait_increment=20, checkpoint=None, **kwargs):
    '''
    Grab the ClassyFire results from many queries in query_dict and preserve order of results of queries and compounds within queries.
    
    Arguments:
        query_dict - dictionary with query ids submitted as keys, and list of compounds submited as values
            type == dict
            eg. if two queries of two compounds each were submited:
                {3526547: ['<SMILES1>', '<SMILES2>'],
                 3526548: ['<SMILES3>', '<SMILES4>']}
                 
         max_wait - maximum amount of idle wait time to conduct while waiting for results still processing
             type == int
             D 100
             
         wait_increment - incriment of time to wait before retrying results retrieval while waiting for processing
             type == int
             
         checkpoint - save point for previous query dict return , otherwise, start from beginning
             type == dict w/ keys "query_completion", "ClassyFying", "queries_classified"
             D None
    
    '''
    assert type(query_dict) == dict, 'Query dict should be a dictionary of keys as query ids, and values as species lists'
    assert isinstance(max_wait, (int, float)), 'max_wait should be a number'
    assert isinstance(wait_increment, (int, float)), 'wait_increment should be a number'
    
    ## resume from checkpoint if specified
    if checkpoint:
        assert type(checkpoint) == dict, 'checkpoint should be a dict dumped with three items: query_completion, ClassyFying, queries_classified'
        
        query_completion = checkpoint['query_completion']
        ClassyFying = checkpoint['ClassyFying']
        queries_classified = checkpoint['queries_classified']
        print('Resuming from checkpoint. State of queries: ', query_completion)
        
    else:
    
        ## set all queries to default not done
        query_completion = {query_id: 'Processing' for query_id in query_dict.keys()}
        ClassyFying = any(status != 'Done' for status in query_completion.values())
        queries_classified = {query_id: None for query_id in query_dict.keys()}
    
    
    ## set up try to save progress on failure
    try:
    
        ## continue loop while all queries are still running
        wait_counter=0
        while ClassyFying:

            ## start the session       
            ## go through each query individually
            for ind, (query_id, status) in enumerate(query_completion.items()):
                print('\n\nQuery ID {} internal status: {}'.format(query_id, status))
                #. Update the status of this query if it is still processing
                if status == 'Processing':
                    print("\tLoading query: {}".format(query_id))
                    try:
                        result = json.loads(get_results(query_id))
                    ## if 500 error
                    except (rq.ReadTimeout, rq.ConnectionError, rq.HTTPError):
                        #. report error
                        print('\tInternal 500 server error returned or timeout on query {}'.format(query_id))
                        #. record unclassified
                        query_classified = {entity_no+1: 'Unclassified' for entity_no in range(len(query_dict[query_id]))}   
                        queries_classified[query_id] = query_classified
                        query_completion[query_id] = 'Failed'

                        #. move to next query
                        continue

                    print('\tClassification status: {}'.format(result['classification_status']))
                    # If it is done, extract it, else continue to next query
                    if result['classification_status'] == 'Done':

                        ## parse the errors, classified and unclassified from result
                        #. start empty of length number of species submitted
                        #. key is the the number associated with the species at the end of the query
                        query_classified = {entity_no+1: 'Unclassified' for entity_no in range(len(query_dict[query_id]))}   

                        #. print report on invalid entities
                        for invalid in result['invalid_entities']:
                            print('\t\tCould not classify SMILES <{}>, report: {} '.format(
                                    invalid['structure'], invalid['report']))

                        #. get classified and unclassified compounds
                        for entity in result['entities']:
                            # will only work for successfully classified
                            try:
                                query_classified[int(entity['identifier'].split('-')[1])] = entity
                            except KeyError:
                                print('\t\tCould not classify SMILES <{}>, report: {} '.format(
                                    entity['smiles'], entity['report']))


                        ## recording the classified queries
                        queries_classified[query_id] = query_classified

                        ## update the status
                        query_completion[query_id] = 'Done'
                        print('\tRecorded classification from query id {}.'.format(query_id))


                    ## if the maximum wait time has been allotted,
                    ## grab classifications even if not all entities are done
                    elif max_wait < wait_counter*wait_increment:

                        print('Max wait time already passed, recording query.')

                        ## parse the errors, classified and unclassified from result
                        #. start empty of length number of species submitted
                        #. key is the the number associated with the species at the end of the query
                        query_classified = {entity_no+1: 'Unclassified' for entity_no in range(len(query_dict[query_id]))}   

                        #. print report on invalid entities
                        for invalid in result['invalid_entities']:
                            print('\t\tCould not classify SMILES <{}>, report: {} '.format(
                                    invalid['structure'], invalid['report']))

                        #. get classified and unclassified compounds
                        for entity in result['entities']:
                            # will only work for successfully classified
                            try:
                                query_classified[int(entity['identifier'].split('-')[1])] = entity
                            except KeyError:
                                print('\t\tCould not classify SMILES <{}>, report: {} '.format(
                                    entity['smiles'], entity['report']))


                        ## add the recording the the classified queries
                        queries_classified[query_id] = query_classified

                        ## update the status
                        query_completion[query_id] = 'Timeout'
                        print('\tRecorded classification from query id {}.'.format(query_id))
                    else:
                        print('\tQuery not ready, passing.')

                else:
                    print('\tAlready recieved, skipping.')

            ## update the tracking of all queries
            ClassyFying = any(status == 'Processing' for status in query_completion.values())
            #. print the queries that are still processing
            print('Queries still processing: ', [query_id for query_id in query_dict.keys()
                                                 if query_completion[query_id] == 'Processing'])


            ## wait a bit if some still working
            if ClassyFying:
                print('Waiting {} s for classification.'.format(str(wait_increment)))
                print('###############################')
                time.sleep(wait_increment)
                wait_counter += 1


        ## prepare the list of completed queries for return
        classified_compound_list = []
        for idx, (query_id, query_classified) in enumerate(queries_classified.items()):
            classified_compound_list.extend(query_classified.values())
            
    except:
        print('Error encountered, saving query_completion, ClassyFying, queries_classified to gcf.pkl')
        
        gcf = {'query_completion': query_completion, 
              'ClassyFying': ClassyFying,
              'queries_classified': queries_classified}
        
        with open('gcf.pkl', 'wb') as handle:
            pickle.dump(gcf, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            
        raise
        
        
    ## save results
    gcf = {'query_completion': query_completion, 
          'ClassyFying': ClassyFying,
          'queries_classified': queries_classified}

    with open('gcf.pkl', 'wb') as handle:
        pickle.dump(gcf, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                
    return queries_classified, classified_compound_list


def dataframe_query_and_update(df, structure_key, **kwargs):
    '''
    From a dataframe containing a column of structures to classify, submit queries and and add retrieved ClassyFire results to the dataframe.
    
    Arguments:
        df - pandas dataframe that contains compounds to be classified
            type == pd.DataFrame
            
        structure_key - column name in df that contains compound structures to submit
            type == str
            
    Returns:
        df_out - pandas dataframe, a copy of the input dataframe with ClassyFire results appended to each index with column 'ClassyFire_output'
            type == pd.DataFrame
            
        queries_classified - dictionary of dictionaries, keys are query IDs submitted, values are dictionaries of ordered entities and their ClassyFire results
            type == dict
            eg. if one query with query id "QID" was submitted with 3 compounds and "results" as ClassyFire API output:
                {QID: 
                    {1: <results1>, 
                     2: <results2>, 
                     3: <results3>} 
                }
    kwargs - keyward arguments
        for query_dict_return_patient: 'max_wait', 'wait_increment'
    '''
    assert type(df) == pd.core.frame.DataFrame, 'df must be a pandas dataframe'
    assert type(structure_key) == str, 'structure_key should be the column name in df with structure'
    
    ## query the database
    query_ids = dataframe_query(df, structure_key, **kwargs)
    #. remove query dict from kwargs - it has been updated by the line above
    kwargs.pop('query_dict', None)
    
    ## get the return classification
    queries_classified, classified_compound_list = query_dict_return_patient(query_ids, **kwargs)
      
    df_out = df.copy()
    ## add to dataframe
    df_out['ClassyFire_output'] = classified_compound_list
    
    return df_out, queries_classified

def expand_ClassyFied_df(df_in, label = 'ClassyFire_output'):
    '''
    Expand the compacted ClassyFire output column in a pandas dataframe into multiple labeled columns, eg. class and superclass. 
    
    Arguments:
        df_in - dataframe containing CLassyFire output column
            type == pd.DataFrame
            
        label - label for the column containing ClassyFire full output
            type == str
    Returns:
        df_out - dataframe with expanded columns appended, inclusing CLassyFire kingdom, class
            type == pd.DataFrame
    '''
    assert type(df_in) == pd.core.frame.DataFrame, 'df_in must be a pandas dataframe'
    assert label in df_in.columns, 'Label column is not in the input dataframe'
    
    df_out = df_in.copy()
    
    ## expand dictionary of dictionaries from direct output
    expanded_column = df_out[label].apply(pd.Series)
#     print(expanded_column)
#     expanded_column.drop(0, axis = 1, inplace=True)

    ## total discription captures all sub description, drop these for kingdom, class etc. for useful columns
    def get_class(row):
        class_name = []
        
        ## iterate through each column (kingdom, etc)
        for class_lvl in row:
            try:
                ## if it has a classification, add it to list
                class_name.append(class_lvl['name'])
                
            except TypeError:
                ## if not classified to that level leave empty
                class_name.append(None)
                
        return class_name
    
    ## apply this to the dataframe producing a dataframe of classes
    expanded_column[['kingdom', 'superclass', 'class', 'subclass']] = \
        expanded_column[['kingdom', 'superclass', 'class', 'subclass']].apply(
            get_class,
            axis=0, raw=True)
    
    ## replace column names for distinguishing of source
    new_columns = list(expanded_column.columns)
    for i, c in enumerate(new_columns):
        new_columns[i] = 'cf_'+c
        
    ## add to the final df and return
    df_out[new_columns] = expanded_column
    
    return df_out