start=`date +%s`
runtime=$((end-start))
data=$(<text_docs_30.json)
url=http://192.168.0.26:5000/cls_json
curl -i \
-H "Accept: application/json" \
-H "Content-Type:application/json" \
-X POST --data $data \
$url
end=`date +%s`
runtime=$((end-start))
echo "runtime $runtime seconds"