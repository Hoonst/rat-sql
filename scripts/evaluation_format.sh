bs="6"
loss="label_smooth"
qv_link="false"
dist_relation="true"
orthog="false"

exp_name="bs=$bs,loss=$loss,qv_link=$qv_link,dist=$dist_relation,orthog=$orthog"
python3 evaluation_format.py --experiment_name=${exp_name}