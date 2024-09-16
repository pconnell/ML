import pandas as pd, numpy as np
import requests

#use another API to get the top 5 lenders listed in CNN article

#uses the global legal entity identifier GLEIF API 
# to get the right organizations from the HMDA database
#and so we can efficiently and effectively name data files
lei_ep = "https://api.gleif.org/api/v1/lei-records"

target_organizations = [
    'NAVY FEDERAL CREDIT UNION',
    'Wells Fargo Bank\\, National Association',
    'ROCKET MORTGAGE\\, LLC',
    'Bank of America\\, National Association',
    #'First National Bank of America',
    #'JP Morgan Chase'
    'JPMorgan Chase Bank\\, National Association'
]

replacer = {
    'NAVY FEDERAL CREDIT UNION':'Navy Federal Credit Union',
    'Wells Fargo Bank\\, National Association': 'Wells Fargo',
    'ROCKET MORTGAGE\\, LLC': 'Rocket Mortgage',
    'Bank of America\\, National Association':'Bank of America',
    #'First National Bank of America',
    'JPMorgan Chase Bank\\, National Association': 'JP Morgan'
}

lei_param = {
    'page[size]':100,
    'page[number]':1,
    'filter[entity.names]':''
}

entities = []

for org in target_organizations:
    print(org)
    lei_param['filter[entity.names]'] = org
    resp = requests.get(lei_ep,lei_param)
    num_pages = resp.json()['meta']['pagination']['lastPage']
    for i in range(1,num_pages+1):
        print(i/num_pages)
        lei_param['page[number]'] = i
        try:
            resp = requests.get(lei_ep,lei_param).json()
            for item in resp['data']:
                lei = item['attributes']['lei']
                name = item['attributes']['entity']['legalName']['name']
                if len(item['attributes']['entity']['otherNames']) > 0:
                    otherName = item['attributes']['entity']['otherNames'][0]['name']
                else:
                    otherName = ''
                #special case - Rocket Mortgage didn't have a BIC, and the
                #correct LEIs for the other lenders did have a BIC, so 
                #accounted for the special case here
                if item['attributes']['bic'] or replacer[org]=='Rocket Mortgage':
                    entities.append({
                        'lei':lei,
                        'name':name,
                        'otherName':otherName,
                        'company':replacer[org]
                    })
        except:
            pass

#build a data frame of the legal entity identifiers for 
#future reference and use within research activities
lei_list = pd.DataFrame.from_records(entities)
lei_list.drop_duplicates(inplace=True)
lei_list.to_csv('../data/lei_list2.csv',index=False)


#############################2ND API########################################
#Now we have all the LEIs...we can use this to go forward and 
#extract the records from HMDA for each entity
#using the home mortgage disclosure act API
hmda_endpoint = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
hmda_endpoint_params = {
    'years':'',
    'loan_types':1, #conventional loans only
    'leis':'' #will change/update based upon what orgs we're downloading
}

years = [2022,2023]

for org in replacer.values():
    hmda_endpoint_params['leis'] = lei_list[
        lei_list['company']==org
    ]['lei'].to_list()
    print(org,hmda_endpoint_params['leis'])
    for year in years:
        hmda_endpoint_params['years'] = year
        resp = requests.get(
            hmda_endpoint,hmda_endpoint_params
        )        
        with open(
            '../data/{}_{}_conventional.csv'.format(
                org,year
            ),
            'wb'
        ) as f:
            f.write(resp.content)
            f.close()


##### INITIAL CLEANING ######

#COLLAPSING EXTRA COLUMNS

data_files = []

to_collapse=[
    'applicant_race-',
    'co-applicant_race-',
    'applicant_ethnicity-',
    'co-applicant_ethnicity-',
    'aus-', 
    'denial_reason-'
]

for file in data_files:
    data = pd.read_csv('')
    for col in to_collapse:
        if col =='denial_reason-':
            v = 5
        else:
            v = 6
        cols = [col+'{}'.format(i) for i in range(1,v)]
        data[col[:-1]] = data[cols].apply(
            lambda x: "','".join(x.dropna().astype(int).astype(str)),
            axis=1
        )
        data[col[:-1]] = "'"+data[col[:-1]] + "'"
        data.drop(columns=cols,inplace=True)
    data.to_csv(''+'_clean.csv',index=False)

## Removing unneeded / unnecessary columns...
"""
derived_loan_product_type
derived_dwelling_category?
derived_msa-md

submission of application
initially payable to institution
applicant / co-applicant above age 62

"""


## Removing unneeded or problematic rows...
#eliminating exempt rows?
###


## Converting columns to categorical...
""" 
loan_type,
loan_purpose,
lien_status
reverse_mortgage
open-end_line_of_credit
business_or_commercial_purpose
hoepa_status
negative_amortization
interest_only_payment #?
balloon_payment #?
other_nonamortizing_features #?
construction_method
occupancy_type
manufactured_home_secured_property_type
manufactured_home_land_property_interest
applicant_credit_score_type
co-applicant_credit_score_type
"""


from sklearn.preprocessing import (
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler
)



#ordinals - 
    # % level 