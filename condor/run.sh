#!/bin/bash

PREFIX=$1
WORKDIR=`pwd`
WEAVER_PATH="${WORKDIR}/weaver-benchmark/weaver"

# Download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda_install.sh
bash miniconda_install.sh -b -p ${WORKDIR}/miniconda
export PATH=$WORKDIR/miniconda/bin:$PATH
pip install numpy pandas scikit-learn scipy matplotlib tqdm PyYAML
pip install numba
pip install uproot3 awkward0 awkward lz4 xxhash
pip install weaver-core
pip install tables
pip install onnxruntime-gpu
pip install tensorboard
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA environment setup
export PATH=$PATH:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.2/lib64

# Clone weaver-benchmark
git clone -b GloParT --recursive https://github.com/zichunhao/weaver-benchmark.git
# ln -s ../top_tagging weaver-benchmark/weaver/top_tagging
cd "${WEAVER_PATH}"
mkdir output


# Training
lr='5e-4'
PREFIX=ak8_MD_inclv8_part_splitreg_addltphp_wmeasonly_manual.useamp.large_fc2048.gm5.ddp-bs256-lr${lr}
config=${WEAVER_PATH}/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/bond/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=0,1,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
${WEAVER_PATH}/train.py \
--train-mode hybrid -o three_coll True -o loss_split_reg True -o loss_gamma 5 \
-o fc_params '[(2048,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr $lr --num-epochs 50 --optimizer ranger --fetch-step 0.00666667 \
--backend nccl \
--data-train \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config ${WEAVER_PATH}/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix ${WEAVER_PATH}/model/${PREFIX}/net \
--log-file ${WEAVER_PATH}/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output ${WEAVER_PATH}/predict/$PREFIX/pred.root

# Predicting
NGPUS=1
python ${WEAVER_PATH}/train.py --predict --gpus 1 \
--train-mode hybrid -o three_coll True -o loss_split_reg True -o loss_gamma 5 \
-o fc_params '[(2048,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 1e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.00666667 \
--data-test \
'xww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*0.root' \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
'ofcttbarfl:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/*.root' \
'ofcttbarsl:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config ${WEAVER_PATH}/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix ${WEAVER_PATH}/model/${PREFIX}/net_epoch-26_state.pt \
--log-file ${WEAVER_PATH}/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output ${WEAVER_PATH}/predict/$PREFIX/pred.root

[ -d "runs/" ] && tar -caf output.tar output/ runs/ || tar -caf output.tar output/