# Install all required packages
import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

# Import packages
import itertools
from census import Census
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

# Connect census package with census API key
# NOTE: To use the Census API, you need to sign-up at this site https://api.census.gov/data/key_signup.html
# You will receive your API key in your email. Copy and paste it here
c = Census("6c30242c40faab9091e343eba633f256c7eecfda")

# Import table that associates county with CA100 defined regions
REGIONAL_MAPPING_FILENAME = 'region_county_mapping.csv'
REGIONAL_MAPPING_FILEPATH = os.path.abspath(REGIONAL_MAPPING_FILENAME)
county_codes_df = pd.read_csv(REGIONAL_MAPPING_FILEPATH)
county_codes_df = county_codes_df.dropna(axis=1)
county_codes_df['FIPS Code'] = county_codes_df['FIPS Code'].astype(str)
county_codes_df['FIPS Code'] = county_codes_df['FIPS Code'].str[1:]

county_codes = county_codes_df.copy().set_index('County').to_dict('index')

# Data scope table
# Set-up what census variables we want to pull
CA_FIPS = '06'
BASE_CODE = 'B01001'
MAX_AGE_CAT = 31
ETH = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
ETH_CODES = [ BASE_CODE + eth_code for eth_code in ETH]
ETH_AGE_CODES = [
    [ eth_code + '_' + str(age_cat).zfill(3) + 'E' for age_cat in range(1,MAX_AGE_CAT+1)]
    for eth_code in ETH_CODES
]
ETH_AGE_CODES = list(itertools.chain.from_iterable(ETH_AGE_CODES))
COUNTIES = list(county_codes.keys())

# Set-up empty table to put data from Census API into
# This creates a table that has number of people per ethnicity, age group, and county with additional labeling of region
# This is based on the regional definition csv and the census variables we want to download
county_census_df = county_codes_df.copy()
county_census_df = county_census_df.reindex(columns=['County', 'Region', 'FIPS Code'] + ETH_AGE_CODES)

# Download Census Data
codes = ETH_AGE_CODES

for i, eth_age_code in enumerate(codes):
    eth_age_data = []
    threads = []
    print("Started " + eth_age_code)
    with ThreadPoolExecutor(max_workers=5) as executor:
        for county in COUNTIES:
            threads.append(executor.submit(c.acs5.state_county,
                                           eth_age_code,
                                           CA_FIPS,
                                           county_codes[county]['FIPS Code']))
        for task in threads:
            eth_age_data.append(task.result()[0][eth_age_code])
    print(eth_age_code + " done. " + "{percent:.3}%".format(percent=((i+1)/len(codes) * 100)) + " done.")
    county_census_df[eth_age_code] = eth_age_data
    county_census_df.to_csv('county_eth_age_breakdown.csv')

# Set County as the index
county_census_df = pd.read_csv('county_eth_age_breakdown.csv')
county_census_df.set_index('County', inplace=True)
county_census_df = county_census_df.drop(county_census_df.columns[[0]], axis=1)

# Identify age groups that are only 2 years in length so all age groups are the same length (4 years)
# Combine 006, 007 (male) 021, 022 (female) cols
ETH_AGE_CODES_MERGED = [
    [ eth_code + '_' + str(age_cat).zfill(3) + 'E' for age_cat in [6,7,21,22]]
    for eth_code in ETH_CODES
]
ETH_AGE_CODES_MERGED = list(itertools.chain.from_iterable(ETH_AGE_CODES_MERGED))
ETH_AGE_CODES_MERGED_ZIPPED = list(zip(ETH_AGE_CODES_MERGED[0::2], ETH_AGE_CODES_MERGED[1::2]))
merged_demo_cols = pd.DataFrame()
for merged_codes in ETH_AGE_CODES_MERGED_ZIPPED:
    merged_codes_list = list(merged_codes)
    merged_col_name = merged_codes[0]+'_merged'
    merged_demo_cols[merged_col_name] = county_census_df[merged_codes_list].sum(axis=1)
# Identify age groups that are only 4 years in length instead of 8 years
# Split 011, 012, 013, 014, 015 (male) 026, 027, 028, 029, 030 (female) cols into a and b
SPLIT_COL_NAMES = [11,12,13,14,15,26,27,28,29,30]
ETH_AGE_CODES_SPLIT = [
    [ eth_code + '_' + str(age_cat).zfill(3) + 'E' for age_cat in SPLIT_COL_NAMES]
    for eth_code in ETH_CODES
]
ETH_AGE_CODES_SPLIT = list(itertools.chain.from_iterable(ETH_AGE_CODES_SPLIT))
split_demo_cols = pd.DataFrame()
for split_code in ETH_AGE_CODES_SPLIT:
    split_demo_cols[split_code+"_a"] = county_census_df[split_code]/2
    split_demo_cols[split_code+"_b"] = county_census_df[split_code]/2

# Perform the actual merging and splitting of age brackets
county_census_viz_df = county_census_df.copy()
county_census_viz_df = county_census_viz_df.drop(columns=ETH_AGE_CODES_MERGED)
county_census_viz_df = county_census_viz_df.drop(columns=ETH_AGE_CODES_SPLIT)
county_census_viz_df = pd.concat([county_census_viz_df, merged_demo_cols, split_demo_cols], axis='columns')
list(county_census_viz_df.columns)

# Identify columns by gender
MIN_MALE_INDEX = 3
MAX_MALE_INDEX = 16
MIN_FEMALE_INDEX = 18
MAX_FEMALE_INDEX = 31

# Identify columns by ethnicity
ETH_CODES_DICT = {
    'A': 'White',
    'B': 'Black or African American',
    'C': 'American Indian and Alaska Native',
    'D': 'Asian',
    'E': 'Native Hawaiian and Other Pacific Islander',
    'F': 'Other Race',
    'G': 'Two or More Races',
    'H': 'White, Not Hispanic or Latino',
    'I': 'Hispanic or Latino',
}

# Each ethnicity has a color. Change this dictionary if you want to change the color.
ETH_COLOR_CODES_DICT = {
    'A': '#1abc9c',
    'B': '#f1c40f',
    'C': '#2ecc71',
    'D': '#e67e22',
    'E': '#3498db',
    'F': '#e74c3c',
    'G': '#fd79a8',
    'H': '#34495e',
    'I': '#95a5a6',
}

# Identify columns by age brackets
AGE_RANGE_DICT = {
    '003': 'Under 5 Years',
    '004': '5 to 9 Years',
    '005': '10 to 14 Years',
    '006': '15 to 17 Years',
    '007': '18 to 19 Years',
    '008': '20 to 24 Years',
    '009': '25 to 29 Years',
    '010': '30 to 34 Years',
    '011': '35 to 44 Years',
    '012': '45 to 54 Years',
    '013': '55 to 64 Years',
    '014': '65 to 74 Years',
    '015': '75 to 84 Years',
    '016': '85 Years and Over',
    '018': 'Under 5 Years',
    '019': '5 to 9 Years',
    '020': '10 to 14 Years',
    '021': '15 to 17 Years',
    '022': '18 to 19 Years',
    '023': '20 to 24 Years',
    '024': '25 to 29 Years',
    '025': '30 to 34 Years',
    '026': '35 to 44 Years',
    '027': '45 to 54 Years',
    '028': '55 to 64 Years',
    '029': '65 to 74 Years',
    '030': '75 to 84 Years',
    '031': '85 Years and Over',
}

AGE_RANGE_TO_INDEX_DICT = {
    'Under 5 Years': 0,
    '5 to 9 Years': 1,
    '10 to 14 Years': 2,
    '15 to 17 Years': 3,
    '18 to 19 Years': 3,
    '20 to 24 Years': 4,
    '25 to 29 Years': 5,
    '30 to 34 Years': 6,
    '35 to 44 Years': 7,
    '45 to 54 Years': 9,
    '55 to 64 Years': 11,
    '65 to 74 Years': 13,
    '75 to 84 Years': 15,
    '85 Years and Over': 17,
}
AGE_RANGES = [
    'Under 5 Years',
    '5 to 9 Years',
    '10 to 14 Years',
    '15 to 19 Years',
    '20 to 24 Years',
    '25 to 29 Years',
    '30 to 34 Years',
    '35 to 39 Years',
    '40 to 44 Years',
    '45 to 49 Years',
    '50 to 54 Years',
    '55 to 59 Years',
    '60 to 64 Years',
    '65 to 69 Years',
    '70 to 74 Years',
    '75 to 79 Years',
    '80 to 84 Years',
    '85 Years and Over',
]

MERGED_AGES = ['021', '022', '006', '007']
SPLIT_AGES = ['011', '012', '013', '014', '015', '026', '027', '028', '029', '030']

# Replaces index in string
def replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + 1:]

# Helper function to get left columns based on census code
def get_left_cols(code):
    eth_code = code[6:7]
    gender_age = code[8:11]

    if eth_code == 'H':
        return 0

    # Eth Order: H, I, B, C, D, E, F, G
    eth_order = ['H', 'I', 'B', 'C', 'D', 'E', 'F', 'G']

    eth_order = eth_order[:eth_order.index(eth_code)]
    eth_offset_cols = [replacer(code, eth_idx, 6)[:12] for eth_idx in eth_order]

    if gender_age in ['021', '022', '006', '007']:
        complementary_age = ''
        if gender_age == '021':
            complementary_age = '022'
        elif gender_age == '022':
            complementary_age = '021'
        elif gender_age == '006':
            complementary_age = '007'
        elif gender_age == '007':
            complementary_age = '006'
        complementary_code = replacer(replacer(code, complementary_age, 8), 'E', 11)
        additional_eth_offset_cols = [replacer(complementary_code, eth_idx, 6)[:12] for eth_idx in eth_order]
        eth_offset_cols = additional_eth_offset_cols + eth_offset_cols
    return eth_offset_cols

# Helper function to get left offset based on census code (return int)
def get_left_offset(code, region):
    gender_age = code[8:11]
    # Get columns to the left
    eth_offset_cols = get_left_cols(code)
    if eth_offset_cols == 0:
        return 0

    # Sum values to get left offset
    regional_df = county_census_df[county_census_df['Region'] == region]

    if set(eth_offset_cols).issubset(regional_df.columns.tolist()):
        if gender_age in ['011', '012', '013', '014', '015', '026', '027', '028', '029', '030']:
            return regional_df.loc[:,eth_offset_cols].sum().sum()/2
        return regional_df.loc[:,eth_offset_cols].sum().sum()
    return eth_offset_cols

# Helper function to translate census code to attribute (gender, age, ethnicity)
def get_demo_col(code, region):
    gender_age = code[8:11]
    eth_code = code[6:7]
    suffix = code[13:] if len(code) > 12 else ''
    suffix_val = 1 if suffix == 'b' else 0

    gender = 'Male' if ((int(gender_age) >= MIN_MALE_INDEX) & (int(gender_age) <= MAX_MALE_INDEX)) else 'Female'
    gender_idx = 0 if gender == 'Male' else 1
    age = AGE_RANGE_DICT[gender_age]
    age_idx = AGE_RANGE_TO_INDEX_DICT[age] + suffix_val
    eth = ETH_CODES_DICT[eth_code]
    eth_left_len = get_left_offset(code, region)

    return {
        'label': {'gender': gender, 'age': age, 'eth': eth},
        'chart': (gender_idx, age_idx, eth_left_len)
    }


census_codes_list = county_census_viz_df.columns[2:].tolist()

# Remove codes we don't want to look at
# Age: 001, 002, 017
# Eth: A
codes_to_remove = []
for code in census_codes_list:
    gender_age = code[8:11]
    eth_code = code[6:7]
    if (eth_code == 'A') | (gender_age == '001') | (gender_age == '017') | (gender_age == '002'):
        print(gender_age, eth_code)
        codes_to_remove.append(code)

for remove_code in codes_to_remove:
    census_codes_list.remove(remove_code)

# Generate Regional Age Pyramid
def regional_age_pyramid(region):
    fig, axes = plt.subplots(figsize=(12,6.5), facecolor='#eaeaf2', ncols=2, sharey=True)
    for census_code in census_codes_list:
        demo_details = get_demo_col(census_code, region)
        gender_index, age_index, eth_left_off = demo_details['chart']
        eth_code = census_code[6:7]
        region_df = county_census_viz_df[county_census_df['Region'] == region]

        # Get region code value
        region_code = census_code[:12]
        region_age = region_code[8:11]
        if region_age in MERGED_AGES:
            region_code = census_code[:12] + '_merged'
        if region_age in SPLIT_AGES:
            region_code = census_code[:12] + '_a'
        region_code_val = region_df[region_code].sum()
        axes[gender_index].barh(age_index,
                                region_code_val,
                                left=eth_left_off,
                                color=ETH_COLOR_CODES_DICT[eth_code],
                                label=ETH_CODES_DICT[eth_code]
                                )

    # Set Titles
    fig.suptitle(region + ' Demographic Pyramid')
    axes[0].set_title('Men Population', fontsize=18, pad=15, zorder=10)
    axes[1].set_title('Women Population', fontsize=18, pad=15, zorder=10)

    # Editing ticks
    axes[0].set(yticks=list(range(0,len(AGE_RANGES))), yticklabels=AGE_RANGES)
    axes[0].yaxis.tick_left()
    axes[0].tick_params(axis='x', rotation=30)
    axes[1].tick_params(axis='x', rotation=-30)

    # Flip men to be the left chart
    axes[0].invert_xaxis()

    # Adjust subplots
    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)

    # Set men and women axis to be the longer length of the two
    # EX: if men is 50,000 and women is 30,000, choose 50,000 as the x lim
    print(axes[0].get_xlim(), axes[1].get_xlim())
    if axes[0].get_xlim()[0] > axes[1].get_xlim()[1]:
        axes[1].set_xlim([0, axes[0].get_xlim()[0]])
    else:
        axes[0].set_xlim([0, axes[1].get_xlim()[1]])

    # Remove zero from one of the axis
    xticks = axes[1].xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)

    # Legend
    # axes[0].legend()
    handles, labels = axes[0].get_legend_handles_labels()
    n = 7
    unique_handles = handles[n - 1::n][:8]
    unique_labels = labels[n-1::n][:8]
    axes[1].legend(unique_handles, unique_labels,
               loc='center left', bbox_to_anchor=(1, 0.5)
               )
    box1 = axes[0].get_position()
    box2 = axes[1].get_position()
    axes[0].set_position([box1.x0, box1.y0, box1.width * 0.65, box1.height])
    axes[1].set_position([box1.width * 1.12, box2.y0, box2.width * 0.65, box2.height])

    filename = 'mpl-bidirectional-' + region
    plt.savefig(filename+'.png', facecolor='#eaeaf2')


regions = county_census_viz_df['Region'].unique().tolist()
for region in regions:
    regional_age_pyramid(region)

