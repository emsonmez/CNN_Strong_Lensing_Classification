import pandas as pd
import requests
from astroquery.mast import Observations

# 1. Scrape the CASTLeS table page (e.g. using pandas.read_html)
ques = requests.get("https://lweb.cfa.harvard.edu/castles/")
tables = pd.read_html(ques.text)
lens_table = tables[0]  # may require indexing adjustment

# 2. Filter for grade A/B
lens_table = lens_table[lens_table['G'].isin(['A','B', 'C'])]

records = []
for idx, row in lens_table.iterrows():
    name = row['LENS NAME']
    ra = row['RA']
    dec = row['DEC']
    z_l = row['z_l']
    z_s = row['z_s']
    records.append({'name': name, 'ra': ra, 'dec': dec, 'z_l': z_l, 'z_s': z_s})

df = pd.DataFrame(records)

# 3. For each system, query MAST for HST observations
for i, rec in df.iterrows():
    obs = Observations.query_region(rec['ra'], rec['dec'], radius=1.5*u.arcsec,
                                     project="HST")
    # choose only WFPC2 or NICMOS exposures
    hst = obs[obs['instrument_name'].isin(['WFPC2','NIC2','NIC1'])]
    files = Observations.get_product_list(hst)
    local = Observations.download_products(files, productSubGroupDescription='FLT', mrp_only=False)