#!/usr/bin/env bash

# Download Transfuser++
wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip
unzip pretrained_models.zip
rm pretrained_models.zip

# Download CIL++ vanilla
wget --content-disposition --trust-server-names 'https://drive.usercontent.google.com/download?export=download&confirm=t&id=1GLo5mVrmyNsb5pLqksYnjR8fN1-ZptHE'
tar -xvf _results.tar.gz
cp ./Ours/Town12346_5/checkpoints/CILv2_multiview_attention_40.pth pretrained_models/CIL/checkpoint.pth
rm -rf ./Ours
rm _results.tar.gz

# Copy CIL attention from server
cp /datafast/121-2/Experiments/dporres/VisionTFM/_results/CILv2_recrexp/CILv2_3cam_Town01_14hdata_AttentionLossKL_AreaDownsampling_notBinarized_20xAttLoss_bs120/checkpoints/CIL_multiview_80.pth pretrained_models/CILattention/checkpoint.pth