import pandas as pd, numpy as np

from sklearn.preprocessing import OrdinalEncoder

import scipy.stats as stats

data_dir = '../data/'
orgs = [
    'Navy Federal Credit Union',
    'Wells Fargo',
    'Rocket Mortgage',
    'Bank of America',
    'JP Morgan'
]
to_collapse=[
    'applicant_race-',
    'co-applicant_race-',
    'applicant_ethnicity-',
    'co-applicant_ethnicity-',
    'aus-', 
    'denial_reason-'
]
yrs = [2022,2023]

#used to perform binary encoding of 29 columns down to 6.
collapser = {
    'applicant_race-':{
         1:0b0000000000000000001,
         2:0b0000000000000000010,
        21:0b0000000000000000100,
        22:0b0000000000000001000,
        23:0b0000000000000010000,
        24:0b0000000000000100000,
        25:0b0000000000001000000,
        26:0b0000000000010000000,
        27:0b0000000000100000000,
         3:0b0000000001000000000,
         4:0b0000000010000000000,
        41:0b0000000100000000000,
        42:0b0000001000000000000,
        43:0b0000010000000000000,
        44:0b0000100000000000000,
         5:0b0001000000000000000,
         6:0b0010000000000000000,
         7:0b0100000000000000000,
         8:0b1000000000000000000
    },
    'co-applicant_race-':{
         1:0b0000000000000000001,
         2:0b0000000000000000010,
        21:0b0000000000000000100,
        22:0b0000000000000001000,
        23:0b0000000000000010000,
        24:0b0000000000000100000,
        25:0b0000000000001000000,
        26:0b0000000000010000000,
        27:0b0000000000100000000,
         3:0b0000000001000000000,
         4:0b0000000010000000000,
        41:0b0000000100000000000,
        42:0b0000001000000000000,
        43:0b0000010000000000000,
        44:0b0000100000000000000,
         5:0b0001000000000000000,
         6:0b0010000000000000000,
         7:0b0100000000000000000,
         8:0b1000000000000000000
    },
    'applicant_ethnicity-':{
        1:0b000000001,
        11:0b000000010,
        12:0b000000100,
        13:0b000001000,
        14:0b000010000,
        2:0b000100000,
        3:0b001000000,
        4:0b010000000,
        5:0b100000000
    },
    'co-applicant_ethnicity-':{
        1:0b000000001,
        11:0b000000010,
        12:0b000000100,
        13:0b000001000,
        14:0b000010000,
        2:0b000100000,
        3:0b001000000,
        4:0b010000000,
        5:0b100000000
    },
    'aus-':{
        1:0b00000001,
        2:0b00000010,
        3:0b00000100,
        4:0b00001000,
        5:0b00010000,
        7:0b00100000,
        6:0b01000000,
        1111:0b10000000,
    }, 
    'denial_reason-':{
        1:0b0000000001,
        2:0b0000000010,
        3:0b0000000100,
        4:0b0000001000,
        5:0b0000010000,
        6:0b0000100000,
        7:0b0001000000,
        8:0b0010000000,
        9:0b0100000000,
        10:0b1000000000
    }
}

#used to identify / filter any columns with invalid
#values based on encoding
collapser_invalids = {
    'applicant_race':{
        'intervals':[(65536,131072)],
        'gt':131072
    },
    'co-applicant_race':{
        'intervals':[(65536,131072),(131072,262144)],
        'gt':262144
    },
    'applicant_ethnicity':{
        'intervals':[(64, 128)],
        'gt':128
    },
    'co-applicant_ethnicity':{
        'intervals':[(64,128), (128,256)],
        'gt':256
    },
    'aus':{
        'intervals':[(64,128)],
        'gt':128
    },
    'denial_reason':{
        'intervals':[],
        'gt':512
    }
}

for org in orgs:

    #read in the file for the lender
    data = pd.read_csv(data_dir+org+'_2023_conventional.csv')

    #rare to have all of 99 columns be the same - drop duplicates
    data.drop_duplicates(inplace=True)

    #use to filter the housing units down - mixed data type
    data['total_units'] = data['total_units'].astype(str)
    #1-4 unit homes only 
    data.drop(
        index=data.loc[~data['total_units'].isin(['1','2','3','4'])].index,
        axis=0,
        inplace=True
    )
    # print(len(data))
    #first lien only 
    data.drop(
        index=data.loc[data['lien_status']!=1].index,
        axis=0,
        inplace=True
    )
    #conforming loans only
    # print(len(data))
    data.drop(
        index=data.loc[data['conforming_loan_limit']!='C'].index,
        axis=0,
        inplace=True
    )
    #not for business or commercial purpose
    # print(len(data))
    if 2 in data['business_or_commercial_purpose']:
        data.drop(
            index=data.loc[data['business_or_commercial_purpose']!=2].index,
            axis=0,
            inplace=True
        )
    # print(len(data))
    # primary residence 
    data.drop(
        index=data.loc[data['occupancy_type']!=1].index,
        axis=0,
        inplace=True
    )

    #for the purpose of purchasing a home
    data.drop(
        index=data.loc[data['loan_purpose']!=1].index,
        axis=0,
        inplace=True
    )

    #provide reasons for dropping these rows here and in main doc.
    data.drop(
        index=data[
            (data['county_code'].isna()) #| 
            #(data['loan_term'].isna()) | 
            #(data['loan_to_value_ratio'].isna()) | #maybe adjust?
            #(data['property_value'].isna()) #maybe fill with median column?
        ].index,
        inplace=True,
        axis=0
    )

    data.drop(
        index=data[
            data['state_code'].isna()
        ].index,
        inplace=True,
        axis=0
    )

    #remove unnecessary columns
    data.drop(
        labels=[
            'activity_year', #only one year
            'derived_msa-md', #not sure if needed or if it is redundant...
            #'state_code', #less detailed, maybe need, maybe don't.
            'census_tract', #do we eliminate or keep?
            'derived_loan_product_type', #not needed ? justify
            'derived_dwelling_category', #not needed ? justify
            'conforming_loan_limit', #one type by scope definition
            'lien_status', #one type by scope definition
            'reverse_mortgage', #results in only one type, eliminate
            'business_or_commercial_purpose', #one type by scope definition
            'negative_amortization', #only different on 11 rows; not worth it.
            'occupancy_type', #one type by scope definition
            'construction_method', #only going to use site-built - scope limits count to approx 7300 manufactured homes
                'manufactured_home_secured_property_type', #need to justify
                'manufactured_home_land_property_interest', #need to justify
            'submission_of_application', #maybe?
            'initially_payable_to_institution', #?
            'derived_ethnicity', #contained in collapsed column - all ethnic data
            'derived_race', #contained in collapsed column
            'loan_type', #defined in scope, i think (conventional)
            'prepayment_penalty_term', #not sure...
            'applicant_age_above_62', #available in other columns
            'co-applicant_age_above_62', #available in other columns
            'total_points_and_fees', #little to no data available in entire dataset
            'rate_spread', #not sure eliminating this is justified...yet.
            'multifamily_affordable_units',# justify?
            #'total_units' #defined within scope of search of being 1-4 units - or should we keep?
        ],
        axis=1,
        inplace=True
    )

    #binary encoding of 4-5 column fields
    for col in collapser.keys():
        #denial reason has 4 columns, others have 5
        if col == 'denial_reason-':
            v=5
        else:
            v=6
        #build a list of column names
        cols=[col+'{}'.format(i) for i in range(1,v)]
        #for each of those columns 
        for c in cols:
            #find the binary encoding; update the column
            #set to 0 if the column is blank
            data[c] = [collapser[col].get(x,0b0) for x in data[c]]
        #add all the columns together to have a single encoded column
        data[col[:-1]] = sum([data[c] for c in cols])
        #drop the 4 to 5 other columns
        data.drop(columns=cols,inplace=True)

    #filter out any invalid binary results
    for col,vals in collapser_invalids.items():
        #if the value exists in an invalid interval - delete it.
        for interv in vals['intervals']:
            data.drop(
                index=data.loc[
                    (data[col] > interv[0]) &
                    (data[col] < interv[1])
                ].index,
                axis=0,
                inplace=True
            )

        #if the value is greater than the max valid value - delete it
        data.drop(
            index=data.loc[
                data[col] > vals['gt']
            ].index,
            axis=0,
            inplace=True
        )

    #fill in instances of NA for loan_to_value with loan amount * 100 / property_value
    inds = data.loc[data['loan_to_value_ratio'].isna()].index
    data.loc[inds,'loan_to_value_ratio'] = data.loc[inds,'loan_amount'] * 100 / data.loc[inds,'property_value']

    #add feature for approval / denial outcome of an application
    data['outcome'] = np.select(
        [
            #counting preapprovals, originations, and loan purchases as approvals
            #in addition to general approvals
            data['action_taken'].isin([1,2,6,8]),
            #denials are preapproval denials as well as regular denials
            data['action_taken'].isin([3,7])
        ],
        [True,False],
        #when the value is 4 or 5, we exclude them
        default=np.nan
    )

    #eliminate where there is no outcome
    data.drop(
        index=data.loc[
            data['outcome'].isna()
        ].index,
        axis=0,
        inplace=True
    )

    #perform ordinal encoding of values.
    OrdEnc = OrdinalEncoder()
    ords = {
        'applicant_age':[[
            '<25',
            '25-34',
            '35-44',
            '45-54',
            '55-64',
            '65-74',
            '>74',
            '8888'
        ]],
        'co-applicant_age':[[
            '<25',
            '25-34',
            '35-44',
            '45-54',
            '55-64',
            '65-74',
            '>74',
            '8888',
            '9999'
        ]],
        'debt_to_income_ratio':[[
            '<20%',
            '20%-<30%',
            '30%-<36%',
            '36',
            '37',
            '38',
            '39',
            '40',
            '41',
            '42',
            '43',
            '44',
            '45',
            '46',
            '47',
            '48',
            '49',
            '50%-60%',
            '>60%',
            'NA',
            'Exempt'
        ]]
    }

    for k,v in ords.items():
        OrdEnc = OrdinalEncoder(categories=v,handle_unknown='use_encoded_value',unknown_value=np.nan)
        t = OrdEnc.fit_transform(np.array(data[[k]].astype(str)))
        data[k] = t


    data.to_csv(
        '../data/{}_2023_clean.csv'.format(org),
        index=False
    )


#combine all frames to single frame
JP = pd.read_csv('../data/JP Morgan_2023_clean.csv')
BoA = pd.read_csv('../data/Bank of America_2023_clean.csv')
WF = pd.read_csv('../data/Wells Fargo_2023_clean.csv')
NFCU = pd.read_csv('../data/Navy Federal Credit Union_2023_clean.csv')
RM = pd.read_csv('../data/Rocket Mortgage_2023_clean.csv')
fr = pd.concat([JP,BoA,WF,NFCU,RM])
lenders = pd.read_csv(
    '../data/lei_list2.csv'
)

#add the lender information - just the name
fr = fr.merge(
    right=lenders[['lei','company']],how='left',on='lei'
)

#get the stats for loan information - median by state
tab = fr.groupby(
    by=['state_code','company'],as_index=False #other loan qualitites? 
).agg({
    'total_loan_costs':np.median,
    'origination_charges':np.median,
    'discount_points':np.median,
    'lender_credits':np.median,
    'interest_rate':np.median,
    'intro_rate_period':stats.mode,
    'debt_to_income_ratio':stats.mode
})

print(tab.head())
print(fr.columns)
#the results of stats.mode produces a tuple, in a list.
#extract only the value of the mode.
tab['intro_rate_period'] = tab['intro_rate_period'].apply(lambda x: x[0][-1])
tab['debt_to_income_ratio'] = tab['debt_to_income_ratio'].apply(lambda x: x[0][-1])
for col in [ 
    'total_loan_costs',
    'origination_charges',
    'discount_points',
    'lender_credits',
    'interest_rate'
]:
    tab[col].fillna(1,inplace=True)
for col in [
    'intro_rate_period',
    'debt_to_income_ratio'
]:
    tab[col].fillna(0,inplace=True)

# tab[[
#     'total_loan_costs',
#     'origination_charges',
#     'discount_points',
#     'lender_credits',
#     'interest_rate',
# ]].fillna(1,inplace=True)

# tab[[
#     'intro_rate_period',
#     'debt_to_income_ratio'
# ]].fillna(0,inplace=True)

#for each column on loan data
for col in [
    'total_loan_costs',
    'origination_charges',
    'discount_points',
    'lender_credits',
    'intro_rate_period',
    'debt_to_income_ratio',
    'interest_rate'
]:
    #for each row in the aggregated stats
    for j, row in tab.iterrows():
        #find where there are blanks for the current row from 
        #the aggregate data and a loan was approved
        i = fr.loc[
            ((fr[col].isnull())|(fr[col].isna())) &
            (fr['state_code']==row['state_code']) &
            (fr['company']==row['company'])
        ].index
        #pythonprint('num_indices:{}\nreplace_val:{}'.format(len(i),row[col]))
        #set blanks equal to the value from the aggregate data
        #whether median or mode
        if np.isnan(row[col]):
            print("{}/{}: {} is an NA value in this case...".format(row['state_code'],row['company'],col))
        fr.loc[i,col] = row[col]

    # fr[col]=fr.apply(
    #     lambda row: row[col] if not np.isnan(row[col]) and row['action_taken']!=3 else tab.loc[row['state_code']==tab['state_code'],col].iloc[0],
    #     axis=1        
    # )

#fill in loan values with 0 where a loan wasn't given / approved
"""
for c in [
    'interest_rate','total_loan_costs','origination_charges',
    'discount_points','lender_credits','intro_rate_period',
    'debt_to_income_ratio'
]:
    #get the indexes of denied loans with blank values
    i = fr.loc[
        (fr[c].isnull()) &
        (fr['action_taken']==3)
    ].index
    #set to zero
    fr.loc[i,c] = 0"""


tab2 = fr.groupby(
    by=['state_code','total_units'],
    as_index=False
).agg({
    'property_value':np.median,
    'loan_to_value_ratio':np.median
})

print(tab2.isnull().sum())

for c in [
    'property_value',
    'loan_to_value_ratio'
]:
    for j, row in tab2.iterrows():
        i = fr.loc[
            (fr[c].isnull()) &
            (fr['state_code']==row['state_code']) &
            (fr['total_units']==row['total_units'])
        ].index
        fr.loc[i,c] = row[c]

#where income is blank - fill in with median family income.

fr['income_from_median'] = fr['income'].isna()

fr['income'] = np.select(
    [fr['income'].isna()],
    [fr['ffiec_msa_md_median_family_income']/1000],
    fr['income']
)

fr['income'] = np.abs(fr['income'])

print(fr['income'].describe())
print(len(fr[fr['income']<0]))

fr.drop(
    labels=[
        'lei',
        #'state_code',
        'hoepa_status',
        'loan_purpose',
    ],
    axis=1,
    inplace=True
)

fr.to_csv('../data/final_clean2.csv',index=False)