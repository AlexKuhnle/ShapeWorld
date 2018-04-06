echo "agreement:"

echo "  existential"
python generate.py -d examples/test1 -U -t agreement -n existential -i 1 -M -H --v1
echo "  relational"
python generate.py -d examples/test1 -U -t agreement -n relational -i 1 -M -H --v1
echo "  quantification"
python generate.py -d examples/test1 -U -t agreement -n quantification -i 1 -M -H --v1
echo "  quantification_complex"
python generate.py -d examples/test1 -U -t agreement -n quantification_complex -i 1 -M -H --v1
echo "  logical"
python generate.py -d examples/test1 -U -t agreement -n logical -i 1 -M -H --v1

echo "  existential (python2)"
python2 generate.py -d examples/test1 -U -t agreement -n existential -i 1 -M -H --v1


echo "classification:"

echo "  shape"
python generate.py -d examples/test1 -U -t classification -n shape -i 1 -M -H --v1

echo "  shape (python2)"
python2 generate.py -d examples/test1 -U -t classification -n shape -i 1 -M -H --v1


echo "generation modes:"

echo "  generate from loaded"
python generate.py -d examples/test1 -t agreement -n existential -s 2,1,3 -i 10 -M -T -Y --v1
python generate.py -d examples/test2 -t agreement -n existential -c examples/test1 -s 1,2,3 -i 10 -M -Y --v1
echo "  generate mixer"
python generate.py -d examples/test1 -t agreement -n existential,relational -s 3,2,1 -i 10 -M -Y --v1
echo "  generate mixer from loaded"
python generate.py -d examples/test1 -t agreement -n relational -s 1,1,1 -i 10 -M -T -Y --v1
python generate.py -d examples/test2 -t agreement -n existential,relational -c examples/test1 -s 3,2,1 -i 10 -M -Y --v1
echo "  generate from config"
python generate.py -d examples/test2 -c examples/test1/agreement-existential+relational.json -s 1,2,3 -i 10 -M -Y --v1
echo "  generate mixer from configs"
python generate.py -d examples/test2 -c examples/test1/agreement-existential.json,examples/test1/agreement-existential+relational.json -s 1,2,3 -i 10 -M -T -Y --v1
echo "  train from previously generated data"
python train.py -t agreement -n existential -c examples/test1 -m cnn_bow -b 2 -i 2 -e 2 -f 2 --v1
python train.py -t agreement -n relational -c examples/test1 -m cnn_bow -b 2 -i 2 -e 2 -f 2 -T --v1
python train.py -t agreement -n existential+relational -c examples/test1 -m cnn_bow -b 2 -i 2 -e 2 -f 2 --v1
python train.py -t agreement -n existential+existential+relational -c examples/test2 -m cnn_bow -b 2 -i 2 -e 2 -f 2 -T --v1


rm -r examples/test1
rm -r examples/test2
