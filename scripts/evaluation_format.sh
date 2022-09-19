bs="12"
loss="softmax"
qv_link="true"
dist_relation="true"
orthog="false"

exp_name="bs=$bs,loss=$loss,qv_link=$qv_link,dist_relation=$dist_relation,orthog=$orthog"
python3 evaluation_format.py --experiment_name=${exp_name}