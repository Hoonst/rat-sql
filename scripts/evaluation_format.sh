bs="6"
qv_link="false"
dist_relation="true"
orthog="false"
orth_init="false"
bi_way="false"
bi_match="true"

exp_name="bs=$bs,qv_link=$qv_link,dist=$dist_relation,orthog=$orthog,orth_init=$orth_init,bi_way=$bi_way,bi_match=$bi_match"
python3 ratsql/commands/evaluation_format.py --experiment_name=${exp_name}