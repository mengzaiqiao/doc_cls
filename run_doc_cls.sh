start=`date +%s`
runtime=$((end-start))
data=$(<text_docs_10.json)
url=http://localhost:5000/cls_json
curl -i \
-H "Accept: application/json" \
-H "Content-Type:application/json" \
-X POST --data "$data" \
$url
end=`date +%s`
runtime=$((end-start))
echo "10 documents runtime $runtime seconds"

start=`date +%s`
runtime=$((end-start))
data=$(<text_docs_20.json)
url=http://localhost:5000/cls_json
curl -i \
-H "Accept: application/json" \
-H "Content-Type:application/json" \
-X POST --data "$data" \
$url
end=`date +%s`
runtime=$((end-start))
echo "20 documents runtime $runtime seconds"

start=`date +%s`
runtime=$((end-start))
data=$(<text_docs_30.json)
url=http://localhost:5000/cls_json
curl -i \
-H "Accept: application/json" \
-H "Content-Type:application/json" \
-X POST --data "$data" \
$url
end=`date +%s`
runtime=$((end-start))
echo "30 documents runtime $runtime seconds"