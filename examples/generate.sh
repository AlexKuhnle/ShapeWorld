# agreement
rm -r examples/agreement
python generate.py -d examples/agreement/oneshape -U -t agreement -n oneshape -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/multishape -U -t agreement -n multishape -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/spatial -U -t agreement -n spatial -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/relational -U -t agreement -n relational -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/quantification_simple -U -t agreement -n quantification_simple -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/quantification -U -t agreement -n quantification -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/quantification_complex -U -t agreement -n quantification_complex -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/combination -U -t agreement -n combination -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5

# classification
rm -r examples/classification
python generate.py -d examples/classification/oneshape -U -t classification -n oneshape -M -H -i 100
python generate.py -d examples/classification/multishape -U -t classification -n multishape -M -H -i 100
python generate.py -d examples/classification/countshape -U -t classification -n countshape -M -H -i 100

# readme data
rm -r examples/readme
python generate.py -d examples/readme -a tar:bzip2 -t agreement -n multishape -f 3,2,1 -i 100 -M -T
python generate.py -d examples/readme -a tar:bzip2 -t agreement -n spatial -f 3,2,1 -i 100 -M

# generate from loaded
python generate.py -d examples/test1 -t agreement -n multishape -c examples/readme -f 2,1,3 -i 100 -M -T
python generate.py -d examples/test2 -t agreement -n multishape -c examples/test1 -f 1,2,3 -i 100 -M
# generate mixer
python generate.py -d examples/test1 -t agreement -n multishape,spatial -f 3,2,1 -i 100 -M -Y
# generate mixer from loaded
python generate.py -d examples/test2 -t agreement -n multishape,spatial -c examples/readme -f 3,2,1 -i 100 -M -Y
# generate from config
python generate.py -d examples/test2 -c examples/test1/agreement-multishape+spatial.json -f 1,2,3 -i 100 -M -Y
# generate mixer from configs
python generate.py -d examples/test2 -c examples/test1/agreement-multishape.json,examples/test1/agreement-multishape+spatial.json -f 1,2,3 -i 100 -M -T -Y

# train from previously generated data
python train.py -t agreement -n multishape -c examples/readme -m cnn_bow -b 2 -i 2 -e 2 -f 2
python train.py -t agreement -n spatial -c examples/readme -m cnn_bow -b 2 -i 2 -e 2 -f 2
python train.py -t agreement -n multishape+spatial -c examples/test1 -m cnn_bow -b 2 -i 2 -e 2 -f 2
python train.py -t agreement -n multishape+multishape+spatial -c examples/test2 -m cnn_bow -b 2 -i 2 -e 2 -f 2 -T

rm -r examples/test1
rm -r examples/test2
