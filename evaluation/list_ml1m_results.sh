base_dir="results/BERT4rec.ml-1m"
for dir in `ls -1 $base_dir | grep ml | grep 1m  | grep 2022_12`;
do
    echo $dir;
    python3 analyze_experiment_in_progress.py $base_dir/$dir/stdout;
done; 
