preproc_mut.ipynb
- create a list of genes which occur in >=6 of the 8 different gene panels -> 225 genes ('data/crc_raw/cBioPortal_files/gene_list.txt')
- fill mutation df with gene mutations for each sample, for each of the 225 genes data/crc_mutpertreat.csv

preproc_ib.ipynb
- list of kinase inhibitors: results/kib_list.csv
- preprocess the remaining clinical columns from data/crc_raw/CRC_2.0-public_clinical_data/
- add 'Histology Category', 'Histology', and 'Derived Grade or Differentiation of Tumor' from data/crc_raw/cBioPortal_files/data_clinical_patient.txt
- add 'CEA' from /data/crc_raw/CRC_2.0-public_clinical_data/tm_level_dataset.csv'

preproc_regimen.ipynb
- crc_egfr_out.csv: cetuximab and panitumumab
- crc_vegf_out.csv: bevacizumab