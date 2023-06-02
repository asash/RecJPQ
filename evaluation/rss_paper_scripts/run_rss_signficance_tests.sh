for model in GRU4rec Caser Sasrec; do
    echo "testing $model... "
    python3 two_predictions_signficance_test.py --metrics "HIT@10,ndcg@10" --first $1/predictions/$model-continuation-bce.json.gz --second $1/predictions/$model-rss-bce.json.gz 
    python3 two_predictions_signficance_test.py --metrics "HIT@10,ndcg@10" --first $1/predictions/$model-continuation-lambdarank.json.gz --second $1/predictions/$model-rss-lambdarank.json.gz 
    python3 two_predictions_signficance_test.py --metrics "HIT@10,ndcg@10" --first $1/predictions/$model-continuation-bce.json.gz --second $1/predictions/$model-continuation-lambdarank.json.gz 
    python3 two_predictions_signficance_test.py --metrics "HIT@10,ndcg@10" --first $1/predictions/$model-rss-bce.json.gz --second $1/predictions/$model-rss-lambdarank.json.gz 
    
    echo ""
done;
