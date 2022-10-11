bs="6"
qv_link="false"
dist_relation="false"
orthog="false"
orth_init="false"
bi_way="true"
bi_match="true"
dp_link='spacy'
plm='bert'

exp_name="bs=$bs,qv_link=$qv_link,dist=$dist_relation,orthog=$orthog,orth_init=$orth_init,bi_way=$bi_way,bi_match=$bi_match,dp_link=$dp_link,plm=$plm"
python3 ratsql/commands/evaluation_format.py --experiment_name=${exp_name}