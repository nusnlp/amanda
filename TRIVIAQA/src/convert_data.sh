data_dir=$1
out_dir=$2
wikidoc_dir=$3
webdoc_dir=$4
mkdir $out_dir


#for x in web-train web-dev verified-web-dev wikipedia-train wikipedia-dev verified-wikipedia-dev; do
for x in verified-web-dev verified-wikipedia-dev; do
    echo "Converting $x"
    python3.5 utils/convert_to_squad_format.py --triviaqa_file $data_dir/$x.json --squad_file $out_dir/$x.json --wikipedia_dir $wikidoc_dir --web_dir $webdoc_dir

    echo "======================="
done
