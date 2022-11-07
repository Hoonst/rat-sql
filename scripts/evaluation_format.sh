bs="6"
qv_link="false"
dist_relation="true"
orthog="false"
orth_init="false"
bi_way="true"
bi_match="true"
dp_link='false'
plm='electra'
att_seq='MP'
layers='24'

exp_name="bs=$bs,qv_link=$qv_link,dist=$dist_relation,orthog=$orthog,orth_init=$orth_init,bi_way=$bi_way,bi_match=$bi_match,dp_link=$dp_link,plm=$plm,att_seq=$att_seq,layers=$layers"
python3 ratsql/commands/evaluation_format.py --experiment_name=${exp_name}