data_dir=$1
out_dir=$2
#test_data=$5
#glovefile=$4 (path of the 300D glove file)
mkdir $out_dir
stage=0

for x in web-train web-dev verified-web-dev  wikipedia-train wikipedia-dev verified-wikipedia-dev; do
    echo "Tokenizing $x"
    python3.5 tokenization/do_tokenization.py -in=$data_dir/$x.json -out=$out_dir/tokenized-$x.json
done

if [ $stage -le 3 ]; then
    ## Download the vocabulary files
    echo "Obtaining the vocabulary files..."
    #TODO
    ## Alternatively you can follow the src/prep_vocab.py to generate them.

fi

for x in web-train web-dev verified-web-dev; do
    echo "Indexing $x"
    python3.5 prep-data/data_prep_with_char.py -data=$out_dir/tokenized-$x.json -id2w=prep-data/id2word_web.json -id2c=prep-data/id2char_web.json -wr=$out_dir/indexed_$x.json
done

echo "Obtaining the vocabulary files..."
curl -L -o prep-data/id2word_wiki.json https://tinyurl.com/ybdvpxcr/triviaqa/id2word_wiki.json
curl -L -o prep-data/id2word_web.json https://tinyurl.com/ybdvpxcr/triviaqa/id2word_web.json
curl -L -o prep-data/id2char_wiki.json https://tinyurl.com/ybdvpxcr/triviaqa/id2char_wiki.json
curl -L -o prep-data/id2char_web.json https://tinyurl.com/ybdvpxcr/triviaqa/id2char_web.json

for x in wikipedia-train wikipedia-dev verified-wikipedia-dev; do
    echo "Indexing $x"
    python prep-data/data_prep_with_char.py -data=$out_dir/tokenized-$x.json -id2w=prep-data/id2word_wiki.json -id2c=prep-data/id2char_wiki.json -wr=$out_dir/indexed_$x.json
done

# Download the embedding matrix
curl -L -o prep-data/embed_mat_wiki.npy https://tinyurl.com/ybdvpxcr/triviaqa/embed_mat_wiki.npy
curl -L -o prep-data/embed_mat_web.npy https://tinyurl.com/ybdvpxcr/triviaqa/embed_mat_web.npy


#if [ $stage -le 7 ]; then
    ## Alternatively prepare the embedding matrix
#   # echo "Preparing embed mat (Optional)..."
#    python prep-data/prep_embed_mat.py -id2w=data/id2word.json -glove $glovefile -out data/embed_mat_opt.npy
#fi

