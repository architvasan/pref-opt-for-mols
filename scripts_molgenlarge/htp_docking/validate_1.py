import pandas as pd
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
import os

try:
    parser = ArgumentParser()#add_help=False)
    
    parser.add_argument(
        "-s", "--scoresdir", type=Path, required=True, help="scoresdir"
    )
    parser.add_argument(
        "-c", "--config", type=Path, required=True, help="config.json file"
    )

    args = parser.parse_args()
    
    fil_list = sorted(os.listdir(f'{args.scoresdir}'))
    
    df = pd.read_csv(f'{args.scoresdir}/{fil_list[0]}')
    
    for i in range(1, len(fil_list)):
        df = pd.concat([df, pd.read_csv(f'{args.scoresdir}/{fil_list[i]}')])
    
    ### Parameters setting
    import json
     
    # Opening JSON file
    f = open(args.config)
     
    # returns JSON object as 
    # a dictionary
    config = json.load(f)
    
    if len(df)>0.8*config['sample_size']:
        print("OK")
    else:
        print("Not OK")
except:
    print("Not OK")
