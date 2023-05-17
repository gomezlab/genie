preproc_mut.ipynb
- create a list of genes which occur in >=6 of the 8 different gene panels -> 224 genes ('data/crc_raw/cBioPortal_files/gene_list.txt')
- fill mutation df with gene mutations for each sample, for each of the 224 genes data/crc_mutoh_pertreat.csv
- filter cna data to 224 genes -> data/crc_cna_pertreat.csv

preproc_ib.ipynb
- list of kinase inhibitors: results/kib_list.csv
- preprocess the remaining clinical columns from data/crc_raw/CRC_2.0-public_clinical_data/
- add 'Histology Category', 'Histology', and 'Derived Grade or Differentiation of Tumor' from data/crc_raw/cBioPortal_files/data_clinical_patient.txt
- add 'CEA' from /data/crc_raw/CRC_2.0-public_clinical_data/tm_level_dataset.csv'
- output: /data/crc_clin_pub.csv

preproc_regimen.ipynb
- crc_egfr_out.csv: cetuximab and panitumumab
- crc_vegf_out.csv: bevacizumab

combine_data.ipynb
- matches outcome data with mutation, CNA, and clinical data (crc_mutoh_pertreat.csv + crc_cna_pertreat.csv + crc_clin_pub.csv + ib_out.csv)
- for all patients, patient-level outcomes: crc_comb.csv
- for kinase inhibitors, treatment-level outcomes: crc_mut_cna_os.csv, crc_mut_cna_pfs.csv
