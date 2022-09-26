bs="6"
loss="label_smooth"
qv_link="false"
dist_relation="true"
orthog="false"
orth_init="true"
bi_way="true"
bi_match="true"

exp_name="bs=$bs,qv_link=$qv_link,dist=$dist_relation,orthog=$orthog,orth_init=$orth_init,bi_way=$bi_way,bi_match=$bi_match"
python3 evaluation_format.py --experiment_name=${exp_name}