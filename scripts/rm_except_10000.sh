folder="logdir/bert_run/bs\=12\,lr\=7.4e-04\,bert_lr\=1.0e-05\,end_lr\=0e0\,att\=1\,loss\=softmax"
# folder="test_shell"
target_idxs=("model-checkpoint-00010100" 
            "model-checkpoint-00020100" 
            "model-checkpoint-00030100" 
            "model-checkpoint-00040100" 
            "model-checkpoint-00050100" 
            "model-checkpoint-00060100" 
            "model-checkpoint-00070100"
            "log.txt"
            "*.json"
            )

new_target_idx=""

for idx in ${target_idxs[*]}
do
    new_target_idxs+="$folder/$idx "   
done

echo $target_idxs
echo $new_target_idxs

elementIn () {
    # shopt -s nocasematch # Can be useful to disable case-matching
    local e
    for e in "${@:2}"; do [[ "$e" == "$1" ]] && return 0; done
    return 1
}

for entry in $folder/*
do 
    echo $entry
    
    if elementIn $entry ${new_target_idxs[@]}; then
        echo TRUE;
    else
        echo FALSE
        rm $entry
fi
done