# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from google.cloud import storage
import pandas as pd
from io import StringIO

import time
import os
import logging

BUCKET_NAME = "pipeline-stream-population-estimates"

def get_client():
    """
    Get a storage client
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/mikelovesbooks/GSS-Cogs/creds/optimum-bonbon-257411-dab80be7170d.json"
    try:
        storage_client = storage.Client()
    except Exception as e:
        raise Exception("Unable to get storage client. Aborting operation:") from e
        
    return storage_client

storage_client = get_client()
bucket = storage_client.get_bucket(BUCKET_NAME)


# +
# Components & Libraries:
import pandas as pd
import json
from pathlib import Path
import datetime
import numpy as np
from gssutils import pathify

from cachecontrol import CacheControl
from cachecontrol.caches import FileCache
from cachecontrol.heuristics import ExpiresAfter
from requests import Session
session = CacheControl(Session(), cache=FileCache('.cache'), heuristic=ExpiresAfter(days=7))

import re

pd.set_option('display.max_colwidth', -1)

absolute_start = datetime.datetime.now()

# +

# NM_2010_1
str_dataset_id = "NM_2010_1"

print(str_dataset_id)
print('https://www.nomisweb.co.uk/api/v01/dataset/NM_2010_1/geography.def.sdmx.json')

# +
# NOMIS RESTful: Get Parent Level Geography for Dataset:

# Get Geography sets using our ID:
baseURL_Geography = 'https://www.nomisweb.co.uk/api/v01/dataset/' + str_dataset_id + '/geography.def.sdmx.json'
df_Geography = pd.read_json(baseURL_Geography)

# Not sure if we want more than one geographies for this, but sooner or later we will do so 
# we'll pretend that we do for now
geographies_we_want = ["2011 output areas"]
geographies_found = []

for a_code in df_Geography["structure"]['codelists']['codelist'][0]['code']:

    if 'description' not in a_code.keys():
        continue
        
    geog_found_name = a_code['description']['value']
    
    for geography_we_want in geographies_we_want:
        if geog_found_name == geography_we_want:
            # Take the whole code dict, never know what we'll need later
            geographies_found.append(a_code)
                 
# Sanity check
assert len(geographies_found) == len(geographies_we_want), \
    "Aborting, we're missing geographies. Wanted {}, got {}.".format(json.dumps(geographies_we_want, indent=2),
                                                                        json.dumps(geographies_found, indent=2))

df_Geography = pd.DataFrame() # Memory

print("Geographies selected: {}".format(json.dumps(geographies_found, indent=2)))


# +

# Get all the dependant codes for these geographies
codes_what_we_want = []
for a_parent_code_dict in geographies_found:

    base_geography_url = 'https://www.nomisweb.co.uk/api/v01/dataset/' + str_dataset_id + '/geography/' + a_parent_code_dict["value"] + '.def.sdmx.json'
    base_geography_dict = session.get(base_geography_url).json()
    
    assert len(base_geography_dict["structure"]["codelists"]) == 1, "We should only have one codelists being" \
                " returned from this call."
    
    for a_child_code_dict in base_geography_dict["structure"]["codelists"]["codelist"][0]["code"]:
        """
        this is one, I'm taking the 9 digit geography code to save us a job later
        -----
        {
        "annotations": {
            "annotation": [
                {
                    "annotationtext": "2011 output areas",
                    "annotationtitle": "TypeName"
                },
                {
                    "annotationtext": 299,
                    "annotationtitle": "TypeCode"
                },
                {
                    "annotationtext": "E00174208",
                    "annotationtitle": "GeogCode"
                }
            ]
        },
        "parentcode": 1228931073,
        "description": {
            "value": "E00174208",
            "lang": "en"
        },
        "value": 1254265842
        },
        """
        try:
            area = a_child_code_dict["annotations"]["annotation"][0]["annotationtext"]
            if area != "2011 output areas":
                raise Exception("This is supposed to be 2011 output areas, got {}.".format(area))
            
            code = a_child_code_dict["description"]["value"]
            #code = a_child_code_dict["value"]
            
            codes_what_we_want.append(code)
        except Exception as e:
            raise Exception("Failed on", json.dumps(a_child_code_dict)) from e

print("Unique codes identified: ", len(set(codes_what_we_want)))

# +
# Check they seem to be in some sort of order
codes_what_we_want.sort()
print("First 5", codes_what_we_want[:5])
print("Last 5", codes_what_we_want[-5:])

geo_query = "{}...{}".format(codes_what_we_want[0], codes_what_we_want[-1])
print("Query range", geo_query)

# +
# Define the fields that we want
fields_we_want = ["DATE","DATE_NAME","DATE_CODE","DATE_TYPE","DATE_TYPECODE","DATE_SORTORDER",
                  "GEOGRAPHY","GEOGRAPHY_NAME","GEOGRAPHY_CODE","GEOGRAPHY_TYPE","GEOGRAPHY_TYPECODE",
                  "GEOGRAPHY_SORTORDER","GENDER","GENDER_NAME","GENDER_CODE","GENDER_TYPE",
                  "GENDER_TYPECODE","GENDER_SORTORDER","C_AGE","C_AGE_NAME","C_AGE_CODE","C_AGE_TYPE",
                  "C_AGE_TYPECODE","C_AGE_SORTORDER","MEASURES","MEASURES_NAME","OBS_VALUE","OBS_STATUS",
                  "OBS_STATUS_NAME","OBS_CONF","OBS_CONF_NAME","URN","RECORD_OFFSET","RECORD_COUNT"]

fields_we_want_query_str = ",".join(fields_we_want)
print("Fields we're asking for", fields_we_want_query_str)

# +

# note: in a dict for own sanity rather than processing
age_ranges = {
    201: 'Aged 0 to 15',
    202: 'Aged 16+',
    203: 'Aged 16 to 64',
    250: 'Aged 16 to 24',
    204: 'Aged 16 to 17',
    205: 'Aged 18 to 24',
    206: 'Aged 18 to 21',
    207: 'Aged 25 to 49',
    208: 'Aged 50 to 64',
    209: 'Aged 65+'
}
age_range_params = "&c_age="+",".join([str(x) for x in age_ranges.keys()])

output_counter = 0
last_index = 0
final_df = pd.DataFrame()

data_url = 'https://www.nomisweb.co.uk/api/v01/dataset/' + str_dataset_id + '.data.csv?date=latest&geography=' + geo_query + age_range_params + "&select=" + fields_we_want_query_str

stream = session.get(data_url, stream=True).raw
stream.decode_content = True

additionalURL = ''
intRecordController = 0

# If the job is partially progressed, pick up where we left off
if os.path.exists("./progress.txt"):
    with open("./progress.txt", "r") as pfile:
        for line in pfile:
            output_counter = int(line.strip())+1 # one AFTER the last success
            break
    intRecordController = 500000 * output_counter
    last_index = intRecordController
    
print("Starting at chunk {}, offset {}".format(output_counter, intRecordController))

finish = False
# The links below are limited to the first 25,000 cells per call.

while True:
    
    start = datetime.datetime.now()
    additionalURL = '&RecordOffset=' + str(intRecordController)
    concatenatedURL = data_url + additionalURL
    stream = session.get(concatenatedURL, stream=True).raw
    stream.decode_content = True

    dataframe = pd.read_csv(stream, engine='c', na_filter=False)

    if (dataframe.empty):
        print("Uploading final csv")
        finish = True

    frames = [final_df, dataframe]
    final_df = pd.concat(frames)
    
    records_since_write = intRecordController - last_index
    
    if records_since_write >= 499999 or finish == True:
        
        print("starting write at", intRecordController, "rows, at ", datetime.datetime.now())
        try_count = 1
        
        print("{} to {}".format(final_df[:1]["RECORD_OFFSET"], final_df[-1:]["RECORD_OFFSET"]))
        
        while True:
            
            output_name = '{}_CensusPop_LMA_ages.csv'.format(str(output_counter))
            try:
                
                logging.warning("Attempting to create output: {}".format(output_name))
                blob = bucket.blob(output_name)
                f = StringIO()
                final_df.to_csv(f, index=False)
                f.seek(0)
                
                logging.warning("Attempting to upload output: {}".format(output_name))
                try:
                    blob.upload_from_file(f, content_type='text/csv')
                except Exception as e:
                    raise e
                
                logging.warning("Clear dataframe and reset int record controller.")
                last_index = intRecordController
                final_df = pd.DataFrame()
                
                # it worked - leave the loop
                logging.warning("write successful")
                break
            
            except Exception as e:
                
                logging.warning("Handling exception: {}".format(str(e)))
        
                # If we've timed out it means we've ..probably.. lost the client connection
                # due to the timescales involved, so reconnect and try again
                if try_count < 6:
                    logging.warning("Retrying, attempt {}".format(str(try_count)))
                    
                    # Simple backoff
                    logging.warning("Backoff, holding for {} seconds.".format(str(try_count * 10)))
                    time.sleep(try_count * 10) # 10, 20, 30, 40, 50 second pauses
                    
                    logging.warning("Getting new storage client")
                    storage_client = get_client()
                    
                    logging.warning("Getting storage bucket {}".format(BUCKET_NAME))
                    bucket = storage_client.get_bucket(BUCKET_NAME)
                    try_count += 1
                else:
                    # If it's still not working after all that, something is definetly borked
                    raise Exception("Failed on all retries, could not write output {}".format(str(output_counter))) from e
            
        # Hard code a one line file to track chunks previously succesfully written
        # so we have the option to pick up where we left off
        try:
            with open("./progress.txt", "w") as f2:
                f2.write(str(output_counter))
        except Exception as e:
            raise e
        output_counter +=1
        
    #Final chunk written, we're done
    if finish:
        break
    
    intRecordController = intRecordController + 25000
        
# Once we've finished everything, nuke the progress file


# -

print("Completion time for complete script is: ", datetime.datetime.now() - absolute_start)



# +
# https://www.nomisweb.co.uk/api/v01/dataset/NM_30_1.data.csv?geography=2092957698&date=latest&sex=8&item=2&pay=1,5&measures=20100,20701

# +

# https://www.nomisweb.co.uk/api/v01/dataset/NM_2002_1.data.csv?geography=2092957699&date=latest&gender=0&c_age=200,203&measures=20100
