# agreement
rm -r examples/agreement

# existential
python generate.py -d examples/agreement/existential-oneshape -U -t agreement -n existential \
    -c configs/agreement/existential/oneshape.json -i 100 -M -H -G --config-values \
    --world-size 100 --captions-per-instance 5 \
    | tee results.txt

python generate.py -d examples/agreement/existential-full_colfree -U -t agreement -n existential \
    -c configs/agreement/existential/full_colfree.json -i 100 -M -H -G --config-values \
    --world-size 100 --captions-per-instance 5 \
    | tee -a results.txt

python generate.py -d examples/agreement/existential-full -U -t agreement -n existential \
    -c configs/agreement/existential/full.json -i 100 -M -H -G --config-values --world-size 100 \
    --captions-per-instance 5 \
    | tee -a results.txt

python generate.py -d examples/agreement/existential-chinese -U -t agreement -n existential \
    -l chinese -c configs/agreement/existential/chinese.json -i 100 -M -H -G --config-values \
    --world-size 100 --captions-per-instance 5 \
    | tee -a results.txt

# relational
python generate.py -d examples/agreement/relational-spatial_twoshapes -U -t agreement \
    -n relational -c configs/agreement/relational/spatial_twoshapes.json -i 100 -M -H -G \
    --config-values --world-size 100 --captions-per-instance 5 \
    | tee -a results.txt

python generate.py -d examples/agreement/relational-spatial -U -t agreement -n relational \
    -c configs/agreement/relational/spatial.json -i 100 -M -H -G --config-values --world-size 100 \
    --captions-per-instance 5 \
    | tee -a results.txt

python generate.py -d examples/agreement/relational-attribute -U -t agreement -n relational \
    -c configs/agreement/relational/attribute.json -i 100 -M -H -G --config-values \
    --world-size 100 --captions-per-instance 5 \
    | tee -a results.txt

python generate.py -d examples/agreement/relational-comparative -U -t agreement -n relational \
    -c configs/agreement/relational/comparative.json -i 100 -M -H -G --config-values \
    --world-size 100 --captions-per-instance 5 \
    | tee -a results.txt

python generate.py -d examples/agreement/relational-full -U -t agreement -n relational \
    -c configs/agreement/relational/full.json -i 100 -M -H -G --config-values --world-size 100 \
    --captions-per-instance 5 \
    | tee -a results.txt

# selection
python generate.py -d examples/agreement/selection-positive -U -t agreement -n selection \
    -c configs/agreement/selection/positive.json -i 100 -M -H -G --config-values --world-size 100 \
    --captions-per-instance 5 \
    | tee -a results.txt

python generate.py -d examples/agreement/selection-superlative -U -t agreement -n selection \
    -c configs/agreement/selection/superlative.json -i 100 -M -H -G --config-values --world-size 100 \
    --captions-per-instance 5 \
    | tee -a results.txt

python generate.py -d examples/agreement/selection-full -U -t agreement -n selection \
    -c configs/agreement/selection/full.json -i 100 -M -H -G --config-values --world-size 100 \
    --captions-per-instance 5 \
    | tee -a results.txt

# quantification
python generate.py -d examples/agreement/quantification-count -U -t agreement -n quantification \
    -c configs/agreement/quantification/count.json -i 100 -M -H -G --config-values \
    --world-size 100 --captions-per-instance 5 \
    | tee -a results.txt

python generate.py -d examples/agreement/quantification-ratio -U -t agreement -n quantification \
    -c configs/agreement/quantification/ratio.json -i 100 -M -H -G --config-values \
    --world-size 100 --captions-per-instance 5 \
    | tee -a results.txt

python generate.py -d examples/agreement/quantification-full -U -t agreement -n quantification \
    -c configs/agreement/quantification/full.json -i 100 -M -H -G --config-values \
    --world-size 100 --captions-per-instance 5 \
    | tee -a results.txt

python generate.py -d examples/agreement/quantification_complex-full -U -t agreement \
    -n quantification_complex -c configs/agreement/quantification_complex/full.json -i 100 -M -H \
    -G --config-values --world-size 100 --captions-per-instance 5 \
    | tee -a results.txt

# logical
python generate.py -d examples/agreement/logical-existential -U -t agreement -n logical \
    -c configs/agreement/logical/existential.json -i 100 -M -H -G --config-values \
    --world-size 100 --captions-per-instance 5 \
    | tee -a results.txt

python generate.py -d examples/agreement/logical-full -U -t agreement -n logical \
    -c configs/agreement/logical/full.json -i 100 -M -H -G --config-values --world-size 100 \
    --captions-per-instance 5 \
    | tee -a results.txt

# classification
rm -r examples/classification

python generate.py -d examples/classification/shape-single -U -t classification -n shape \
    -c configs/classification/shape/single.json -i 100 -M -H -G --config-values --world-size 100 \
    | tee -a results.txt

python generate.py -d examples/classification/shape-multi -U -t classification -n shape \
    -c configs/classification/shape/multi.json -i 100 -M -H -G \
    | tee -a results.txt

python generate.py -d examples/classification/shape-count -U -t classification -n shape \
    -c configs/classification/shape/count.json -i 100 -M -H -G --config-values --world-size 100 \
    | tee -a results.txt

# readme data
rm -r examples/readme

python -O generate.py -d examples/readme -a tar:bzip2 -t agreement -n existential -v readme -s 3,2,1 \
    -i 100 -M -T \
    | tee -a results.txt

python -O generate.py -d examples/readme -a tar:bzip2 -t agreement -n relational -v readme -s 3,2,1 \
    -i 100 -M \
    | tee -a results.txt
