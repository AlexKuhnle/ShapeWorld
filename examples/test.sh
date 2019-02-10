echo "agreement:"

echo "  existential"
python -O generate.py -d examples/test1 -U -t agreement -n existential -i 1 -M -H
echo "  existential chinese"
python -O generate.py -d examples/test1 -U -t agreement -n existential -l chinese -c configs/agreement/existential/chinese.json -i 1 -M -H
echo "  relational"
python -O generate.py -d examples/test1 -U -t agreement -n relational -i 1 -M -H
echo "  selection"
python -O generate.py -d examples/test1 -U -t agreement -n selection -i 1 -M -H
echo "  quantification"
python -O generate.py -d examples/test1 -U -t agreement -n quantification -i 1 -M -H
echo "  quantification_complex"
python -O generate.py -d examples/test1 -U -t agreement -n quantification_complex -i 1 -M -H
echo "  logical"
python -O generate.py -d examples/test1 -U -t agreement -n logical -i 1 -M -H

echo "  existential (python2)"
python2 -O generate.py -d examples/test1 -U -t agreement -n existential -i 1 -M -H


echo "classification:"

echo "  shape"
python -O generate.py -d examples/test1 -U -t classification -n shape -i 1 -M -H

echo "  shape (python2)"
python2 -O generate.py -d examples/test1 -U -t classification -n shape -i 1 -M -H


echo "generation modes:"

echo "  generate from loaded"
python -O generate.py -d examples/test1 -t agreement -n existential -s 2,1,3 -i 10 -M -T -Y
python -O generate.py -d examples/test2 -t agreement -n existential -c examples/test1 -s 1,2,3 -i 10 -M -Y
echo "  generate mixer"
python -O generate.py -d examples/test1 -t agreement -n existential,relational -s 3,2,1 -i 10 -M -Y
echo "  generate mixer from loaded"
python -O generate.py -d examples/test1 -t agreement -n relational -s 1,1,1 -i 10 -M -T -Y
python -O generate.py -d examples/test2 -t agreement -n existential,relational -c examples/test1 -s 3,2,1 -i 10 -M -Y
echo "  generate from config"
python -O generate.py -d examples/test2 -c examples/test1/agreement-existential+relational.json -s 1,2,3 -i 10 -M -Y
echo "  generate mixer from configs"
python -O generate.py -d examples/test2 -c examples/test1/agreement-existential.json,examples/test1/agreement-existential+relational.json -s 1,2,3 -i 10 -M -T -Y
echo "  train from previously generated data"
python -O train.py -t agreement -n existential -c examples/test1 -m cnn_bow -b 2 -i 2 -e 2 -f 2
python -O train.py -t agreement -n relational -c examples/test1 -m cnn_bow -b 2 -i 2 -e 2 -f 2 -T
python -O train.py -t agreement -n existential+relational -c examples/test1 -m cnn_bow -b 2 -i 2 -e 2 -f 2
python -O train.py -t agreement -n existential+existential+relational -c examples/test2 -m cnn_bow -b 2 -i 2 -e 2 -f 2 -T


rm -r examples/test1
rm -r examples/test2
